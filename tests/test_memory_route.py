from datetime import datetime

from mnemonics.memory_route import duration_event_query, route_memory_query


def test_routes_explicit_relative_event_to_time_target():
    ref = datetime(2023, 3, 25, 23, 37)
    route = route_memory_query("What kitchen appliance did I buy 10 days ago?", reference_time=ref)
    assert route.lanes[0] == "temporal-target"
    assert "personal-state" in route.lanes
    assert route.target_time == datetime(2023, 3, 15, 23, 37)
    assert route.window_days == 1


def test_routes_last_weekday_without_benchmark_labels():
    ref = datetime(2023, 4, 18, 2, 9)  # Tuesday
    route = route_memory_query("Who did I meet with during lunch last Tuesday?", reference_time=ref)
    assert route.lanes[0] == "temporal-target"
    assert route.target_time.date().isoformat() == "2023-04-11"
    assert route.window_days == 0


def test_duration_query_is_not_given_a_fake_target():
    ref = datetime(2023, 6, 30)
    route = route_memory_query("How many weeks ago did I attend the Nordstrom sale?", reference_time=ref)
    assert route.lanes[0] == "temporal-duration"
    assert route.target_time is None


def test_recommendation_routes_to_preference_lane():
    route = route_memory_query("Do you think it would be a good idea to attend my high school reunion?")
    assert route.lanes[0] == "preference"
    assert route.lanes[-1] == "semantic"


def test_personal_state_routes_before_semantic():
    route = route_memory_query("What was my previous occupation?")
    assert route.lanes == ("personal-state", "semantic")


def test_plain_query_falls_back_to_semantic():
    assert route_memory_query("Explain vector databases").lanes == ("semantic",)


def test_duration_event_query_strips_count_wrapper():
    assert duration_event_query(
        "How many weeks ago did I attend the friends and family sale at Nordstrom?"
    ) == "I attend the friends and family sale at Nordstrom"
    assert duration_event_query("How many months have passed since I moved house?") == "I moved house"


def test_duration_event_query_leaves_complex_duration_question_intact():
    q = "How long had I been using the rug when I rearranged the room?"
    assert duration_event_query(q) == q.rstrip("?")
