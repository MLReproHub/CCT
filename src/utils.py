import humanize
from torch.nn import Module


def to_human_readable(number: int, size_format: str = '%.2f', return_number: bool = False) -> str:
    """
    Convert input number to a human-readable string (e.g. 15120 --> 15K)
    Src: https://github.com/achariso/gans-thesis/blob/main/src/utils/string.py
    :param number: the input integer
    :param size_format: format argument of humanize.naturalsize()
    :param return_number: set to True to return input number after human-readable and inside parentheses
    :return: human-readable formatted string
    """
    num_string = humanize.naturalsize(number, format=size_format)
    num_string = num_string.replace('.0', '').replace('Byte', '').replace('kB', 'K').rstrip('Bs').replace(' ', '')
    num_string = num_string.replace('G', 'B')  # billions
    return num_string + (f' ({number})' if return_number else '')


def get_total_params(model: Module, print_table: bool = False, sort_desc: bool = False) -> int or None:
    """
    Get total number of parameters from given nn.Module.
    Src: https://github.com/achariso/gans-thesis/blob/main/src/utils/pytorch.py
    :param model: model to count parameters for
    :param print_table: if True prints counts for every sub-module and returns None, else returns total count only
    :param sort_desc: if True sorts array in DESC order wrt to parameter count before printing
    :return: total number of parameters if $print$ is set to False, else prints counts and returns nothing
    """
    total_count_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = 0
    count_dict = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        count_dict.append({
            'name': name,
            'count': count,
            'count_hr': to_human_readable(count),
            'prc': '%.2f %%' % (count / total_count_orig)
        })
        total_count += count

    assert total_count_orig == total_count, "Should be equal..."

    if print_table is True:
        from prettytable import PrettyTable

        if sort_desc:
            count_dict = sorted(count_dict, key=lambda k: k['count'], reverse=True)

        table = PrettyTable()
        table.field_names = ["Module", "Count", "Count Human", "Percentage"]
        [table.add_row(count_dict[i].values()) for i in range(len(count_dict))]

        print(table.get_string(fields=["Module", "Count Human", "Percentage"]))
        print(f"Total Trainable Params: {to_human_readable(total_count)}")
        return None

    return total_count
