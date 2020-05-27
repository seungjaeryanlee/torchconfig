import pytest

import torchconfig


class TestGetSubdict:
    def test_empty_dict_and_list(self):
        assert len(torchconfig.get_subdict({}, [], ignore_cases=False)) == 0

    def test_empty_dict_and_list_case_insensitive(self):
        assert len(torchconfig.get_subdict({}, [], ignore_cases=True)) == 0

    def test_nonempty_dict_empty_list(self):
        assert len(torchconfig.get_subdict({"A": 1}, [], ignore_cases=False)) == 0

    def test_nonempty_dict_empty_list_case_insensitive(self):
        assert len(torchconfig.get_subdict({"B": 2}, [], ignore_cases=True)) == 0

    def test_one_subdict_key(self):
        d = torchconfig.get_subdict({"C": 3}, ["C"], ignore_cases=False)
        assert len(d) == 1
        assert d["C"] == 3

    def test_one_subdict_key_case_insensitive(self):
        d = torchconfig.get_subdict({"DEF": 4}, ["DeF"], ignore_cases=True)
        assert len(d) == 1
        assert d["DeF"] == 4
        assert "DEF" not in d

    def test_two_subdict_keys(self):
        d = torchconfig.get_subdict({"G": 5, "H": 6}, ["G", "H"], ignore_cases=False)
        assert len(d) == 2
        assert d["G"] == 5
        assert d["H"] == 6

    def test_two_subdict_keys_case_insensitive(self):
        d = torchconfig.get_subdict({"IJ": 5, "K": 6}, ["iJ", "k"], ignore_cases=True)
        assert len(d) == 2
        assert d["iJ"] == 5
        assert d["k"] == 6
        assert "IJ" not in d
        assert "K" not in d

    def test_subdict_key_not_in_original_dict(self):
        d = torchconfig.get_subdict({"L": 7}, ["M"], ignore_cases=False)
        assert len(d) == 0

    def test_subdict_key_not_in_original_dict_case_insensitive(self):
        d = torchconfig.get_subdict({"N": 8}, ["O"], ignore_cases=True)
        assert len(d) == 0


class TestFilterArgs:
    def test_empty_kwargs_no_args(self):
        def no_args_func(): pass
        filtered_kwargs = torchconfig.filter_args({}, no_args_func, ignore_cases=False)
        assert len(filtered_kwargs) == 0

    def test_empty_kwargs_no_args_case_insensitive(self):
        def no_args_func(): pass
        filtered_kwargs = torchconfig.filter_args({}, no_args_func, ignore_cases=True)
        assert len(filtered_kwargs) == 0

    def test_empty_kwargs_one_arg(self):
        def one_arg_func(A): pass
        filtered_kwargs = torchconfig.filter_args({}, one_arg_func, ignore_cases=False)
        assert len(filtered_kwargs) == 0

    def test_empty_kwargs_one_arg_case_insensitive(self):
        def one_arg_func(B): pass
        filtered_kwargs = torchconfig.filter_args({}, one_arg_func, ignore_cases=True)
        assert len(filtered_kwargs) == 0

    def test_nonempty_kwargs_with_args(self):
        def two_args_func(C, D): pass
        filtered_kwargs = torchconfig.filter_args({"D": -1, "E": -2}, two_args_func, ignore_cases=False)
        assert len(filtered_kwargs) == 1
        assert filtered_kwargs["D"] == -1
        assert "C" not in filtered_kwargs
        assert "E" not in filtered_kwargs

    def test_nonempty_kwargs_with_args(self):
        def two_args_func(F, g): pass
        filtered_kwargs = torchconfig.filter_args({"G": -3, "H": -4}, two_args_func, ignore_cases=True)
        assert len(filtered_kwargs) == 1
        assert filtered_kwargs["g"] == -3
        assert "F" not in filtered_kwargs
        assert "H" not in filtered_kwargs

    def test_kwargs_has_identical_keys_case_insensitive(self):
        def no_args_func(): pass
        with pytest.raises(Exception):
            filtered_kwargs = torchconfig.filter_args({"I": -5, "i": -6}, no_args_func, ignore_cases=True)

    def test_func_has_identical_args_case_insensitive(self):
        def two_args_func(J, j): pass
        with pytest.raises(Exception):
            filtered_kwargs = torchconfig.filter_args({}, two_args_func, ignore_cases=True)
