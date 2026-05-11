from src.masking.entity_extraction import EntityExtractor


def test_repeated_entity_positions_are_marked():
    tokens = ["Alice", "saw", "Bob", ".", "Alice", "waved", "to", "Bob"]
    positions = EntityExtractor(min_count=2, top_entities=8).get_entity_positions(tokens)
    assert {0, 2, 4, 7}.issubset(positions)


def test_unrepeated_entities_are_not_marked():
    tokens = ["Alice", "saw", "Bob", "once"]
    positions = EntityExtractor(min_count=2, top_entities=8).get_entity_positions(tokens)
    assert positions == set()
