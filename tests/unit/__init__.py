# Test package marker.
#
# Without it, pytest's default "prepend" import mode imports a file in this
# directory by its bare basename, so two same-named test files in two unmarked
# directories collide and one is silently dropped from the run -- and the suite
# still reports green. Six other test packages here already carry this file;
# tests/unit and tests/ui did not.
