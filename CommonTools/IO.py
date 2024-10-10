import os


def get_sub_dirs(root_dir, end_with=None):
    list_outputs = []

    list_files = os.listdir(root_dir)
    no_sub_dir = True
    for file in list_files:
        cur_path = root_dir + '/' + file
        if os.path.isdir(cur_path):
            no_sub_dir = False
            list_outputs += get_sub_dirs(cur_path, end_with=end_with)

    if no_sub_dir:
        if end_with is None:
            list_outputs.append(root_dir)
        else:
            find_end_with = False
            for file in os.listdir(root_dir):
                if file.find(end_with) > -1:
                    find_end_with = True
                    break
            if find_end_with:
                list_outputs.append(root_dir)

    return list_outputs


def find_files_in_dir(input_dir,
                      must_include_all=None,
                      must_include_one_of=None,
                      must_exclude_all = None
                      ):
    assert must_include_all is None or isinstance(must_include_all, str) or isinstance(must_include_all, list)
    assert must_include_one_of is None or isinstance(must_include_one_of, str) or isinstance(must_include_one_of, list)
    assert must_exclude_all is None or isinstance(must_exclude_all, str) or isinstance(must_exclude_all, list)

    if isinstance(must_include_all, str):
        must_include_all = [must_include_all]
    if isinstance(must_include_one_of, str):
        must_include_one_of = [must_include_one_of]
    if isinstance(must_exclude_all, str):
        must_exclude_all = [must_exclude_all]

    output = []
    for file in os.listdir(input_dir):
        is_target = True
        if must_include_all is not None:
            for target_str in must_include_all:
                assert isinstance(target_str, str)
                if file.lower().find(target_str.lower()) == -1:
                    is_target = False
                    break
        if not is_target:
            continue

        if must_include_one_of is not None:
            is_target = False
            for target_str in must_include_one_of:
                assert isinstance(target_str, str)
                if file.lower().find(target_str.lower()) > -1:
                    is_target = True
                    break
            if not is_target:
                continue

        is_target = True
        if must_exclude_all is not None:
            for target_str in must_exclude_all:
                assert isinstance(target_str, str)
                if file.lower().find(target_str.lower()) > -1:
                    is_target = False
                    break
        if not is_target:
            continue
        output.append(f"{input_dir}/{file}")

    return output
