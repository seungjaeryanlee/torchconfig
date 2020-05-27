import inspect


def get_subdict(original_dict, subdict_keys, ignore_cases=False):
    if ignore_cases:
        # In case dict is not ordered (Python < 3.6)
        keys = list(original_dict.keys())
        lowercase_keys = [k.lower() for k in keys]
        values = [original_dict[k] for k in keys]

        subdict = {}
        for k in subdict_keys:
            if k.lower() in lowercase_keys:
                subdict[k] = values[lowercase_keys.index(k.lower())]
        
        return subdict
    else:
        return {k: v for k, v in original_dict.items() if k in subdict_keys}


def filter_args(all_kwargs, func, ignore_cases=False):
    args = inspect.getfullargspec(func).args
    assert len(args) == len(set(args))
    assert len(all_kwargs.keys()) == len(set(all_kwargs.keys()))
    filtered_kwargs = get_subdict(all_kwargs, args, ignore_cases)
    return filtered_kwargs
