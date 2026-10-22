//! Distant-terrain faces as one 16-byte record each, expanded into their
//! corners by the vertex shader (`terrain.wgsl`'s `vs_far`). A merged LOD face
//! is an axis-aligned rectangle with one material and one light value, so its
//! four 24-byte vertices and six indices (120 bytes of the GPU's memory
//! bandwidth per quad) carried 16 bytes of information. Far terrain is bound
//! by that bandwidth on integrated GPUs, not by its pixels.
//!
//! Layout, in tile-local blocks (horizontal coordinates `0..=16384`, vertical
//! ones offset by 64 so the world's floor is 0; a fluid's top surface sits
//! 1/9 below a whole block, which a flag per edge records):
//!
//! - `w0`: `u0` (15 bits), `v0` (15 bits), direction bits 0-1
//! - `w1`: plane (15 bits), `u1 - u0` (11 bits), direction bit 2, the plane
//!   (Up/Down) or `v0` (sides) lowered, `v1` lowered
//! - `w2`: `v1 - v0` (11 bits), sprite layer (16 bits, with `OPAQUE_LAYER`)
//! - `w3`: RGB colour (24 bits), light (sky in bits 24-27, block in 28-31)
//!
//! Up/Down faces: the plane is a height and `u`, `v` are x and z. North and
//! South: the plane is z, `u` is x and `v` a height. West and East: the plane
//! is x, `u` is z and `v` a height.
use crate::mesher::FaceGroups;
use horizon_assets::model::Direction;

/// What `FarQuad::new` records of a face's vertical extent below the block
/// grid: a fluid's surface (`mesh.rs`'s `Segment`) is this far below its block.
const LOWERED: f32 = 1.0 / 9.0;
/// Heights are stored from the lowest the format holds.
const FLOOR: i32 = -64;
/// The widest face, in blocks (`mesh.rs` splits wider ones).
pub const MAX_FACE: f32 = 1024.0;
const COORDINATE: u32 = (1 << 15) - 1;
const EXTENT: u32 = (1 << 11) - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FarQuad(pub [u32; 4]);

/// A tile's faces: solid ones by facing (`FaceGroups`), then translucent ones.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FarMesh {
    pub solid: Vec<FarQuad>,
    pub faces: FaceGroups,
    pub translucent: Vec<FarQuad>,
}

fn direction_index(direction: Direction) -> u32 {
    match direction {
        Direction::Down => 0,
        Direction::Up => 1,
        Direction::North => 2,
        Direction::South => 3,
        Direction::West => 4,
        Direction::East => 5,
    }
}

const DIRECTIONS: [Direction; 6] = [
    Direction::Down,
    Direction::Up,
    Direction::North,
    Direction::South,
    Direction::West,
    Direction::East,
];

/// A height as stored, and whether it is a fluid's lowered surface.
fn height(y: f32) -> (u32, bool) {
    let block = y.ceil();
    let lowered = block - y > 0.5 * LOWERED;
    debug_assert!(
        !lowered || ((block - y) - LOWERED).abs() < 1e-4,
        "far face height {y} off the block grid"
    );
    let stored = (block as i32 - FLOOR).clamp(0, COORDINATE as i32) as u32;
    (stored, lowered)
}

fn horizontal(x: f32) -> u32 {
    debug_assert_eq!(x.fract(), 0.0, "far face edge {x} off the block grid");
    (x as i32).clamp(0, COORDINATE as i32) as u32
}

fn unheight(stored: u32, lowered: bool) -> f32 {
    (stored as i32 + FLOOR) as f32 - if lowered { LOWERED } else { 0.0 }
}

impl FarQuad {
    /// A face in the plane `plane` covering `u0..u1` by `v0..v1` (see the
    /// module's layout for what the axes are per direction). `light` is a
    /// span's: sky and block light in the high nibble of each byte.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        direction: Direction,
        plane: f32,
        [u0, v0, u1, v1]: [f32; 4],
        layer: u16,
        light: u16,
        color: u32,
    ) -> Self {
        let d = direction_index(direction);
        let vertical_plane = matches!(direction, Direction::Up | Direction::Down);
        let (p, low_a) = if vertical_plane {
            height(plane)
        } else {
            (horizontal(plane), false)
        };
        let u0 = horizontal(u0);
        let du = horizontal(u1).saturating_sub(u0).min(EXTENT);
        let (v0, v1, low_a, low_b) = if vertical_plane {
            (horizontal(v0), horizontal(v1), low_a, false)
        } else {
            let (a, low_a) = height(v0);
            let (b, low_b) = height(v1);
            (a, b, low_a, low_b)
        };
        let dv = v1.saturating_sub(v0).min(EXTENT);
        debug_assert_eq!(light & 0x0f0f, 0, "far light {light:#x} has low bits");
        let light = u32::from((light >> 4) & 15) | (u32::from((light >> 12) & 15) << 4);
        Self([
            u0 | (v0 << 15) | ((d & 3) << 30),
            p | (du << 15) | ((d >> 2) << 26) | (u32::from(low_a) << 27) | (u32::from(low_b) << 28),
            dv | (u32::from(layer) << 11),
            (color & 0xff_ffff) | (light << 24),
        ])
    }
    pub fn direction(self) -> Direction {
        DIRECTIONS[((self.0[0] >> 30) | (((self.0[1] >> 26) & 1) << 2)) as usize]
    }
    pub fn layer(self) -> u16 {
        (self.0[2] >> 11) as u16
    }
    /// The light as a span records it (`Span::light`).
    pub fn light(self) -> u16 {
        let b = self.0[3] >> 24;
        (((b & 15) << 4) | ((b >> 4) << 12)) as u16
    }
    /// RGBA8, opaque, as `TerrainVertex::color`.
    pub fn color(self) -> u32 {
        self.0[3] & 0xff_ffff | 0xff00_0000
    }
    /// The plane and the rectangle `[u0, v0, u1, v1]`, as given to `new`.
    pub fn rect(self) -> (f32, [f32; 4]) {
        let [w0, w1, w2, _] = self.0;
        let (low_a, low_b) = ((w1 >> 27) & 1 != 0, (w1 >> 28) & 1 != 0);
        let u0 = w0 & COORDINATE;
        let v0 = (w0 >> 15) & COORDINATE;
        let p = w1 & COORDINATE;
        let u1 = (u0 + ((w1 >> 15) & EXTENT)) as f32;
        let v1 = v0 + (w2 & EXTENT);
        if matches!(self.direction(), Direction::Up | Direction::Down) {
            (unheight(p, low_a), [u0 as f32, v0 as f32, u1, v1 as f32])
        } else {
            (
                p as f32,
                [u0 as f32, unheight(v0, low_a), u1, unheight(v1, low_b)],
            )
        }
    }
    /// The corners in drawing order, as `vs_far` places them: the same
    /// points and winding a `TerrainVertex` quad of the face had.
    pub fn corners(self) -> [[f32; 3]; 4] {
        let direction = self.direction();
        let (p, [u0, v0, u1, v1]) = self.rect();
        let mut points = match direction {
            Direction::Up | Direction::Down => [[u0, p, v1], [u1, p, v1], [u1, p, v0], [u0, p, v0]],
            Direction::North | Direction::South => {
                [[u0, v1, p], [u1, v1, p], [u1, v0, p], [u0, v0, p]]
            }
            Direction::West | Direction::East => {
                [[p, v1, u1], [p, v1, u0], [p, v0, u0], [p, v0, u1]]
            }
        };
        if matches!(
            direction,
            Direction::Down | Direction::South | Direction::East
        ) {
            points.reverse()
        }
        points
    }
    /// The axis (0 x, 1 y, 2 z) and side the face looks toward, and its plane.
    pub fn facing(self) -> (usize, bool, f32) {
        let plane = self.rect().0;
        match self.direction() {
            Direction::Down => (1, false, plane),
            Direction::Up => (1, true, plane),
            Direction::North => (2, false, plane),
            Direction::South => (2, true, plane),
            Direction::West => (0, false, plane),
            Direction::East => (0, true, plane),
        }
    }
}

impl FarMesh {
    pub fn is_empty(&self) -> bool {
        self.solid.is_empty() && self.translucent.is_empty()
    }
    /// Orders the solid faces by facing (`FaceGroups`) and the translucent
    /// ones with tops last: seen from above, a fluid's sides then lie under
    /// its surface, which is how distant water is seen.
    pub fn finish(&mut self) {
        self.faces = FaceGroups::group_by(&mut self.solid, |quad| quad.facing());
        self.translucent
            .sort_by_key(|quad| quad.direction() == Direction::Up);
    }
    /// The faces as `TerrainVertex` quads with the UVs `vs_far` samples,
    /// for tests that look at corners.
    #[cfg(test)]
    pub fn vertices(quads: &[FarQuad]) -> Vec<crate::mesher::TerrainVertex> {
        quads
            .iter()
            .flat_map(|quad| {
                let points = quad.corners();
                let uvs = points.map(|pos| match quad.direction() {
                    Direction::Up | Direction::Down => [pos[0], -pos[2]],
                    Direction::North | Direction::South => [pos[0], -pos[1]],
                    Direction::West | Direction::East => [pos[2], -pos[1]],
                });
                let corner = [0, 1].map(|i| {
                    uvs.iter()
                        .map(|uv| uv[i])
                        .fold(f32::INFINITY, f32::min)
                        .floor()
                });
                points.into_iter().zip(uvs).map(move |(pos, uv)| {
                    crate::mesher::TerrainVertex::tiled(
                        pos,
                        [uv[0] - corner[0], uv[1] - corner[1]],
                        quad.layer(),
                        quad.color(),
                        quad.light(),
                    )
                })
            })
            .collect()
    }
    /// The mesh as section vertices (`vertices`), for tests.
    #[cfg(test)]
    pub fn section(&self) -> crate::mesher::SectionMesh {
        crate::mesher::SectionMesh {
            solid: Self::vertices(&self.solid),
            faces: self.faces,
            translucent: Self::vertices(&self.translucent),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_round_trip_through_the_record() {
        for direction in DIRECTIONS {
            let vertical = matches!(direction, Direction::Up | Direction::Down);
            let cases: &[(f32, [f32; 4])] = if vertical {
                &[
                    (63.0, [0.0, 0.0, 16.0, 16.0]),
                    (-64.0, [16368.0, 4.0, 16384.0, 1028.0]),
                    (80.0 - LOWERED, [3.0, 5.0, 7.0, 9.0]),
                    (319.0, [0.0, 0.0, 1024.0, 1024.0]),
                ]
            } else {
                &[
                    (16.0, [0.0, 62.0, 16.0, 64.0]),
                    (0.0, [3.0, -64.0, 4.0, 320.0]),
                    (16384.0, [100.0, 63.0 - LOWERED, 1124.0, 70.0]),
                    (5.0, [8.0, 40.0, 9.0, 41.0 - LOWERED]),
                    (5.0, [8.0, 41.0 - LOWERED, 9.0, 42.0 - LOWERED]),
                ]
            };
            for &(plane, rect) in cases {
                let quad =
                    FarQuad::new(direction, plane, rect, 0x8123, 15 << 4 | 7 << 12, 0x12_3456);
                assert_eq!(quad.direction(), direction);
                let (p, r) = quad.rect();
                assert!(
                    (p - plane).abs() < 1e-4,
                    "{direction:?} plane {p} != {plane}"
                );
                for i in 0..4 {
                    assert!(
                        (r[i] - rect[i]).abs() < 1e-4,
                        "{direction:?} {r:?} != {rect:?}"
                    );
                }
                assert_eq!(quad.layer(), 0x8123);
                assert_eq!(quad.light(), 15 << 4 | 7 << 12);
                assert_eq!(quad.color(), 0xff12_3456);
            }
        }
    }

    #[test]
    fn corners_wind_toward_the_face() {
        for direction in DIRECTIONS {
            let quad = FarQuad::new(direction, 8.0, [2.0, 3.0, 5.0, 7.0], 0, 0, 0);
            let [a, b, c, _] = quad.corners().map(glam::Vec3::from);
            let normal = (b - a).cross(c - a).normalize();
            let (axis, positive, _) = quad.facing();
            let mut expected = glam::Vec3::ZERO;
            expected[axis] = if positive { 1.0 } else { -1.0 };
            assert!(
                (normal - expected).length() < 1e-5,
                "{direction:?}: {normal} != {expected}"
            );
        }
    }
}
