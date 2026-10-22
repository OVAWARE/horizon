use super::*;

#[test]
fn identifier_adds_default_namespace() {
    assert_eq!(identifier("stone").unwrap(), "minecraft:stone");
    assert_eq!(identifier("a:b").unwrap(), "a:b");
}

#[test]
fn positive_modulo_wraps_negatives() {
    assert!((positive_modulo(-0.25, 1.0) - 0.75).abs() < 1e-6);
    assert!((positive_modulo(1.5, 1.0) - 0.5).abs() < 1e-6);
}

#[test]
fn display_context_names_round_trip() {
    for name in ["gui", "ground", "fixed", "head"] {
        assert_eq!(DisplayContext::by_name(name).unwrap().name(), name);
    }
}

#[test]
fn plain_models_read_only_the_stack() {
    let node = |v: Value| ItemModelNode::decode(&v).unwrap().reads_only_stack();
    let leaf = |tint: &str| {
        serde_json::json!({
            "type": "minecraft:model",
            "model": "minecraft:item/leather_helmet",
            "tints": [{"type": tint, "default": 0}],
        })
    };
    assert!(node(leaf("minecraft:dye")));
    assert!(node(serde_json::json!({
        "type": "minecraft:composite",
        "models": [leaf("minecraft:dye"), {"type": "minecraft:empty"}],
    })));
    // The owner's team colour, and choices made by the owner or the level.
    assert!(!node(leaf("minecraft:team")));
    assert!(!node(serde_json::json!({
        "type": "minecraft:condition",
        "property": "minecraft:using_item",
        "on_true": leaf("minecraft:dye"),
        "on_false": leaf("minecraft:dye"),
    })));
}
