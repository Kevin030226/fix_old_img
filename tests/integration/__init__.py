# Test package marker.
#
# Without it, pytest's default prepend import mode imports a file in this
# directory by its bare basename, so two same-named test files in two unmarked
# directories collide and one is silently dropped from the run -- and the suite
# still reports green.
#
# tests/unit/test_test_layout.py requires every test package to carry a marker
# that says this, so an empty file cannot be left looking like a stray.
