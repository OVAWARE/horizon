//! Far terrain tiles: compact face records (`lod::quads`) in their own
//! storage arena, expanded into corners by `terrain.wgsl`'s `vs_far` and drawn
//! in the world pass with the terrain's fragment shaders, fog and depth.
//! Tiles are culled with their actual bounds rather than the 16³ boxes of the
//! near-section residency walk.
use super::{Frustum, Renderer, arena::Arena};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::{
        TileKey,
        quads::{FarMesh, FarQuad},
    },
    mesher::FaceGroups,
};
use glam::{DVec3, Vec3};
use std::time::{Duration, Instant};

/// Bytes of one face record.
const QUAD: u64 = std::mem::size_of::<FarQuad>() as u64;
/// An indirect draw: vertex count, instance count, first vertex, first instance.
type Draw = [u32; 4];

/// Finished tile meshes not uploaded yet, the tiles they replace (removed
/// once every mesh is in) and the columns they leave out.
#[derive(Default)]
pub(super) struct Pending {
    meshes: FxHashMap<TileKey, FarMesh>,
    removed: FxHashSet<TileKey>,
    cutout: Option<FxHashSet<(i32, i32)>>,
}

impl Pending {
    /// Takes in the next job, with the same result as applying both in
    /// turn: newer meshes replace waiting ones, tiles meshed again are not
    /// removed, and tiles removed now drop their waiting meshes.
    fn add(
        &mut self,
        meshes: Vec<(TileKey, FarMesh)>,
        removed: Vec<TileKey>,
        cutout: Option<FxHashSet<(i32, i32)>>,
    ) {
        for (key, mesh) in meshes {
            self.removed.remove(&key);
            self.meshes.insert(key, mesh);
        }
        for key in removed {
            self.meshes.remove(&key);
            self.removed.insert(key);
        }
        if cutout.is_some() {
            self.cutout = cutout;
        }
    }
}

/// A resident tile: its records in the far arena, solid ones by facing,
/// then translucent ones.
pub(super) struct Tile {
    page: u32,
    start: u64,
    pub solid: u32,
    pub translucent: u32,
    faces: FaceGroups,
    bounds: Bounds,
}

#[derive(Clone, Copy)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn mesh(mesh: &FarMesh) -> Option<Self> {
        let mut corners = mesh
            .solid
            .iter()
            .chain(&mesh.translucent)
            .flat_map(|quad| quad.corners())
            .map(Vec3::from);
        let first = corners.next()?;
        let mut bounds = Self {
            min: first,
            max: first,
        };
        for pos in corners {
            bounds.min = bounds.min.min(pos);
            bounds.max = bounds.max.max(pos);
        }
        Some(bounds)
    }

    fn visible(self, origin: DVec3, distance: f32, frustum: &Frustum) -> Option<u64> {
        let min = origin + self.min.as_dvec3();
        let max = origin + self.max.as_dvec3();
        let nearest = DVec3::ZERO.clamp(min, max);
        if nearest.x * nearest.x + nearest.z * nearest.z > f64::from(distance).powi(2)
            || !frustum.intersects(min.as_vec3(), max.as_vec3())
        {
            return None;
        }
        Some(((min + max) * 0.5).length_squared().to_bits())
    }
}

/// A tile's instance: its camera-relative origin (`vs_far` reads `xyz`).
pub(super) fn instance(origin: [f32; 3]) -> [f32; 4] {
    [origin[0], origin[1], origin[2], 1.0]
}

pub(super) struct Visible {
    pub key: TileKey,
    pub distance: u64,
    pub origin: [f32; 3],
}

impl Tile {
    pub fn visible(
        &self,
        key: TileKey,
        camera: DVec3,
        distance: f32,
        frustum: &Frustum,
    ) -> Option<Visible> {
        let origin = DVec3::from(key.origin().map(f64::from)) - camera;
        Some(Visible {
            key,
            distance: self.bounds.visible(origin, distance, frustum)?,
            origin: origin.as_vec3().to_array(),
        })
    }
}

/// The far arena, its pipelines and this frame's draws.
pub(super) struct Far {
    arena: Arena,
    layout: wgpu::BindGroupLayout,
    pub solid: wgpu::RenderPipeline,
    pub translucent: wgpu::RenderPipeline,
    /// Per arena page, this frame's binding of it.
    groups: Vec<Option<wgpu::BindGroup>>,
    /// Solid draws front to back, then translucent ones back to front.
    draws: Vec<Draw>,
    pages: Vec<u32>,
    solid_draws: usize,
    indirect: wgpu::Buffer,
    indirect_capacity: u64,
}

impl Far {
    /// `pipeline(layout, label, vertex entry, fragment entry, buffers, blend)`
    /// builds a terrain pipeline (`gfx.rs`'s `pipeline_in`).
    pub fn new(
        device: &wgpu::Device,
        terrain_layout: &wgpu::BindGroupLayout,
        pipeline: impl Fn(
            &wgpu::PipelineLayout,
            &str,
            &str,
            &[Option<wgpu::VertexBufferLayout<'_>>],
            Option<wgpu::BlendState>,
        ) -> wgpu::RenderPipeline,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far terrain faces"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Far terrain layout"),
            bind_group_layouts: &[Some(terrain_layout), Some(&layout)],
            ..Default::default()
        });
        let origin = wgpu::vertex_attr_array![5=>Float32x4];
        let buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &origin,
        })];
        Self {
            arena: Arena::storage(device, QUAD, 1 << 16),
            solid: pipeline(
                &pipeline_layout,
                "Far terrain solid",
                "fs_solid",
                &buffers,
                None,
            ),
            translucent: pipeline(
                &pipeline_layout,
                "Far terrain translucent",
                "fs_translucent",
                &buffers,
                Some(wgpu::BlendState::ALPHA_BLENDING),
            ),
            layout,
            groups: Vec::new(),
            draws: Vec::new(),
            pages: Vec::new(),
            solid_draws: 0,
            indirect: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Far terrain draws"),
                size: 1024 * 16,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            indirect_capacity: 1024,
        }
    }

    pub fn bytes(&self) -> u64 {
        self.arena.bytes()
    }

    pub fn clear(&mut self) {
        self.arena.clear();
    }

    /// This frame's draws of the visible tiles (closest first), whose
    /// instances start at `first_instance`; `eye` is camera-relative.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        uploads: &mut super::upload::Uploads,
        encoder: &mut wgpu::CommandEncoder,
        tiles: &std::collections::HashMap<TileKey, Tile>,
        visible: &[Visible],
        first_instance: usize,
        eye: Vec3,
        multi_draw: bool,
    ) {
        self.draws.clear();
        self.pages.clear();
        for (i, v) in visible.iter().enumerate() {
            let tile = &tiles[&v.key];
            let instance = (first_instance + i) as u32;
            tile.faces
                .ranges(eye - Vec3::from(v.origin), |first, quads| {
                    let first = tile.start + u64::from(first);
                    self.draws
                        .push([quads * 6, 1, (first * 6) as u32, instance]);
                    self.pages.push(tile.page);
                });
        }
        self.solid_draws = self.draws.len();
        for (i, v) in visible.iter().enumerate().rev() {
            let tile = &tiles[&v.key];
            if tile.translucent > 0 {
                let first = tile.start + u64::from(tile.solid);
                self.draws.push([
                    tile.translucent * 6,
                    1,
                    (first * 6) as u32,
                    (first_instance + i) as u32,
                ]);
                self.pages.push(tile.page);
            }
        }
        self.groups.clear();
        for &page in &self.pages {
            let page = page as usize;
            if self.groups.len() <= page {
                self.groups.resize_with(page + 1, || None);
            }
            if self.groups[page].is_none() {
                self.groups[page] = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Far terrain faces"),
                    layout: &self.layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.arena.buffer(page as u32).as_entire_binding(),
                    }],
                }));
            }
        }
        if multi_draw && !self.draws.is_empty() {
            if self.draws.len() as u64 > self.indirect_capacity {
                self.indirect_capacity = (self.draws.len() as u64).next_power_of_two();
                self.indirect = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Far terrain draws"),
                    size: self.indirect_capacity * 16,
                    usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            uploads.buffer(
                encoder,
                &self.indirect,
                0,
                bytemuck::cast_slice(&self.draws),
            );
        }
    }

    pub fn draws(&self) -> usize {
        self.draws.len()
    }

    /// Draws the solid or translucent draws `prepare` made, with the terrain
    /// bindings in group 0 and `instances` (the frame's origins) bound here.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        instances: &wgpu::Buffer,
        translucent: bool,
        multi_draw: bool,
    ) {
        let (mut i, end) = if translucent {
            (self.solid_draws, self.draws.len())
        } else {
            (0, self.solid_draws)
        };
        if i == end {
            return;
        }
        pass.set_pipeline(if translucent {
            &self.translucent
        } else {
            &self.solid
        });
        pass.set_vertex_buffer(0, instances.slice(..));
        while i < end {
            let page = self.pages[i];
            let run = self.pages[i..end]
                .iter()
                .take_while(|&&p| p == page)
                .count();
            pass.set_bind_group(
                1,
                self.groups[page as usize].as_ref().expect("bound far page"),
                &[],
            );
            if multi_draw {
                pass.multi_draw_indirect(&self.indirect, i as u64 * 16, run as u32);
            } else {
                for d in &self.draws[i..i + run] {
                    pass.draw(d[2]..d[2] + d[0], d[3]..d[3] + 1);
                }
            }
            i += run;
        }
    }
}

impl Renderer {
    pub fn resident_lods(&self) -> usize {
        self.lods.len()
    }

    /// Resident tiles per level: (level, tiles, solid quads, translucent quads).
    pub fn lod_levels(&self) -> Vec<(u8, usize, u64, u64)> {
        let mut levels: Vec<(u8, usize, u64, u64)> = Vec::new();
        for (key, tile) in &self.lods {
            let i = match levels.iter().position(|l| l.0 == key.level) {
                Some(i) => i,
                None => {
                    levels.push((key.level, 0, 0, 0));
                    levels.len() - 1
                }
            };
            levels[i].1 += 1;
            levels[i].2 += u64::from(tile.solid);
            levels[i].3 += u64::from(tile.translucent);
        }
        levels.sort_unstable();
        levels
    }

    pub fn upload_lod(&mut self, key: TileKey, mesh: &FarMesh) {
        self.remove_lod(key);
        let Some(bounds) = Bounds::mesh(mesh) else {
            return;
        };
        let quads = (mesh.solid.len() + mesh.translucent.len()) as u64;
        let encoder = self.section_copies.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Section uploads"),
                })
        });
        // With video and system memory both full, the tile is not drawn.
        let Some((page, start)) = self.far.arena.alloc(&self.device, encoder, quads) else {
            return;
        };
        for (offset, part) in [(0, &mesh.solid), (mesh.solid.len(), &mesh.translucent)] {
            if !part.is_empty() {
                let bytes: &[u8] = bytemuck::cast_slice(part);
                self.uploaded_bytes += bytes.len() as u64;
                self.section_copy_bytes += bytes.len() as u64;
                self.section_uploads.buffer(
                    encoder,
                    self.far.arena.buffer(page),
                    (start + offset as u64) * QUAD,
                    bytes,
                );
            }
        }
        if self.section_copy_bytes >= super::SECTION_COPY_LIMIT
            && let Some(copies) = self.take_section_copies()
        {
            self.queue.submit([copies]);
        }
        self.lods.insert(
            key,
            Tile {
                page,
                start,
                solid: mesh.solid.len() as u32,
                translucent: mesh.translucent.len() as u32,
                faces: mesh.faces,
                bounds,
            },
        );
    }

    pub fn remove_lod(&mut self, key: TileKey) {
        if let Some(tile) = self.lods.remove(&key) {
            self.far.arena.release(
                tile.page,
                tile.start,
                u64::from(tile.solid + tile.translucent),
            );
        }
    }

    pub fn clear_lods(&mut self) {
        self.lod_pending = None;
        for key in self.lods.keys().copied().collect::<Vec<_>>() {
            self.remove_lod(key);
        }
        self.set_lod_cutout(Default::default());
    }

    /// Queues a finished mesh job (`Lod::update`, `Lod::removed`,
    /// `Lod::take_cutout`) for `upload_lods`, joining one still uploading
    /// (`Pending::add`).
    pub fn queue_lods(
        &mut self,
        meshes: Vec<(TileKey, FarMesh)>,
        removed: Vec<TileKey>,
        cutout: Option<FxHashSet<(i32, i32)>>,
    ) {
        if meshes.is_empty() && removed.is_empty() && cutout.is_none() {
            return;
        }
        self.lod_pending
            .get_or_insert_default()
            .add(meshes, removed, cutout);
    }

    /// Uploads queued tile meshes for about `budget` (at least one), then,
    /// once all are in, retires the tiles they replace.
    pub fn upload_lods(&mut self, budget: Duration) {
        let start = Instant::now();
        while let Some(pending) = &mut self.lod_pending {
            let Some(&key) = pending.meshes.keys().next() else {
                let pending = self.lod_pending.take().expect("pending tiles");
                for key in pending.removed {
                    self.remove_lod(key);
                }
                if let Some(cutout) = pending.cutout {
                    self.set_lod_cutout(cutout);
                }
                return;
            };
            let mesh = pending.meshes.remove(&key).expect("pending tile");
            self.upload_lod(key, &mesh);
            if start.elapsed() >= budget {
                return;
            }
        }
    }

    /// The chunk columns the resident tiles now leave out (`Lod::take_cutout`).
    /// Removed sections they no longer leave out are released.
    pub fn set_lod_cutout(&mut self, cutout: crate::fast_hash::FxHashSet<(i32, i32)>) {
        self.lod_cutout = cutout;
        let released: Vec<_> = self
            .lod_retiring
            .iter()
            .copied()
            .filter(|key| !self.lod_cutout.contains(&(key.0, key.2)))
            .collect();
        for key in released {
            self.lod_retiring.remove(&key);
            self.release_section(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_assets::model::Direction;

    #[test]
    fn joined_jobs_apply_as_if_in_turn() {
        let key = |x| TileKey { x, z: 0, level: 0 };
        let mut pending = Pending::default();
        // The first job meshes 1 and 2 and retires 3 and 4.
        pending.add(
            vec![(key(1), FarMesh::default()), (key(2), FarMesh::default())],
            vec![key(3), key(4)],
            Some([(0, 0)].into_iter().collect()),
        );
        // The second meshes 3 again and retires 2.
        pending.add(vec![(key(3), FarMesh::default())], vec![key(2)], None);
        let mut meshes: Vec<_> = pending.meshes.keys().map(|k| k.x).collect();
        meshes.sort_unstable();
        let mut removed: Vec<_> = pending.removed.iter().map(|k| k.x).collect();
        removed.sort_unstable();
        assert_eq!((meshes, removed), (vec![1, 3], vec![2, 4]));
        assert!(pending.cutout.is_some_and(|c| c.contains(&(0, 0))));
    }

    #[test]
    fn large_tile_bounds_include_water_and_do_not_cull_by_origin() {
        let mesh = FarMesh {
            solid: vec![
                FarQuad::new(Direction::Up, -64.0, [0.0, 0.0, 1.0, 1.0], 0, 0, 0),
                FarQuad::new(
                    Direction::Up,
                    100.0,
                    [1023.0, 1023.0, 1024.0, 1024.0],
                    0,
                    0,
                    0,
                ),
            ],
            translucent: vec![FarQuad::new(
                Direction::Up,
                120.0,
                [512.0, 512.0, 513.0, 513.0],
                0,
                0,
                0,
            )],
            ..Default::default()
        };
        let bounds = Bounds::mesh(&mesh).unwrap();
        assert_eq!(bounds.min, Vec3::new(0.0, -64.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(1024.0, 120.0, 1024.0));
        let view = Frustum {
            planes: super::super::frustum(
                super::super::world_projection(70.0, 16.0 / 9.0, 8192.0)
                    * crate::camera::view_matrix(0.0, 0.0),
            ),
        };
        // The tile origin is farther than the range, but its far edge reaches
        // the camera. Negative world origins therefore remain visible.
        assert!(
            bounds
                .visible(DVec3::new(-1000.0, -80.0, -512.0), 256.0, &view)
                .is_some()
        );
        assert!(
            bounds
                .visible(DVec3::new(-2000.0, -80.0, -512.0), 256.0, &view)
                .is_none()
        );
        assert!(Bounds::mesh(&FarMesh::default()).is_none());
    }
}
