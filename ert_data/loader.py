import pandas as pd


def data_loader_factory(observation_type):
    """
    Currently, the methods returned by this factory differ. They should not.
    TODO: Remove discrepancies between returned methods.
        See https://github.com/equinor/libres/issues/808
    """
    if observation_type == "GEN_OBS":
        return load_general_data, load_general_obs
    elif observation_type == "SUMMARY_OBS":
        return load_summary_data, load_summary_obs
    elif observation_type == "BLOCK_OBS":
        raise NotImplementedError
        # return load_block_data
    else:
        raise TypeError("Unknown observation type: {}".format(observation_type))


def load_general_data(facade, data_key, case_name):

    time_steps = [
        int(key.split("@")[1])
        for key in facade.all_data_type_keys()
        if facade.is_gen_data_key(key) and data_key in key
    ]
    data = pd.DataFrame()

    for time_step in time_steps:
        gen_data = facade.load_gen_data(case_name, data_key, time_step).T
        data = data.append(gen_data)
    return data


def load_general_obs(facade, observation_keys, case_name):
    observations = []
    for observation_key in observation_keys:
        obs_vector = facade.get_observations()[observation_key]
        data = []
        for time_step in obs_vector.getStepList().asList():
            # Observations and its standard deviation are a subset of the simulation data.
            # The index_list refers to indices in the simulation data. In order to
            # join these data in a DataFrame, pandas inserts the obs/std
            # data into the columns representing said indices.
            # You then get something like:
            #      0   1   2
            # OBS  NaN NaN 42
            # STD  NaN NaN 4.2
            #   0  1.2 2.3 7.5
            node = obs_vector.getNode(time_step)
            index_list = [node.getIndex(nr) for nr in range(len(node))]

            # arrays = [index_list, [i-1 for i in index_list]]
            arrays = [index_list, index_list]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])

            data.append(
                pd.DataFrame(
                    [node.get_data_points()], columns=index, index=["OBS"]
                ).append(
                    pd.DataFrame([node.get_std()], columns=index, index=["STD"])
                )
            )
        data = pd.concat(data, axis=1)
        data = pd.concat({observation_key: data}, axis=1)
        observations.append(data)

    return pd.concat(observations, axis=1)


def _get_block_measured(ensamble_size, block_data):
    data = pd.DataFrame()
    for ensamble_nr in range(ensamble_size):
        data = data.append(pd.DataFrame([block_data[ensamble_nr]], index=[ensamble_nr]))
    return data


def load_summary_data(facade, data_key, case_name):
    args = (facade, data_key, case_name)
    data = _get_summary_data(*args)
    return data


def load_summary_obs(facade, observation_keys, case_name):
    data_key = facade.get_data_key_for_obs_key(observation_keys[0])
    args = (facade, data_key, case_name)
    data = _get_summary_observations(*args)
    obs_map = {}
    for obs_key in observation_keys:
        obs_map[obs_key] = data.pipe(_remove_inactive_report_steps, *(facade, obs_key))
    return pd.concat(obs_map, axis=1)


def _get_summary_data(facade, data_key, case_name):
    data = facade.load_all_summary_data(case_name, [data_key])
    data = data[data_key].unstack(level=-1)
    data = data.set_index(data.index.values)
    return data


def _get_summary_observations(facade, data_key, case_name):
    data = facade.load_observation_data(case_name, [data_key]).transpose()
    # The index from SummaryObservationCollector is {data_key} and STD_{data_key}"
    # to match the other data types this needs to be changed to OBS and STD, hence
    # the regex.
    data = data.set_index(data.index.str.replace(r"\b" + data_key, "OBS", regex=True))
    data = data.set_index(data.index.str.replace("_" + data_key, ""))
    data.columns.name = "key_index"
    return data


def _remove_inactive_report_steps(data, facade, observation_key, *args):
    # XXX: the data returned from the SummaryObservationCollector is not
    # specific to an observation_key, this means that the dataset contains all
    # observations on the data_key. Here the extra data is removed.
    if data.empty:
        return data

    obs_vector = facade.get_observations()[observation_key]
    active_indices = []
    for step in obs_vector.getStepList():
        active_indices.append(step - 1)
    data = data.iloc[:, active_indices]
    arrays = [data.columns.to_list(), active_indices]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])
    data.columns = index
    return data
