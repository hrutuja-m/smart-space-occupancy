from smart_space.tracking.line_counter import LineCounter


def test_entry_and_exit_counting():
    lc = LineCounter(line=((0, 100), (200, 100)), start_count=3)
    lc.update({0: (50, 10)})     # above the line
    lc.update({0: (50, 190)})    # crossed downward -> "in"
    assert lc.entries == 1 and lc.occupancy == 4

    lc.update({0: (50, 10)})     # crossed back up -> "out"
    assert lc.exits == 1 and lc.occupancy == 3


def test_occupancy_never_negative():
    lc = LineCounter(line=((0, 100), (200, 100)), start_count=0)
    lc.update({1: (10, 10)})
    lc.update({1: (10, 190)})     # in
    lc.update({1: (10, 10)})      # out
    lc.update({1: (10, 190)})     # in again... net should stay >= 0
    assert lc.occupancy >= 0


def test_no_crossing_no_count():
    lc = LineCounter(line=((0, 100), (200, 100)))
    for y in (10, 20, 30, 25):
        lc.update({0: (50, y)})
    assert lc.entries == 0 and lc.exits == 0
