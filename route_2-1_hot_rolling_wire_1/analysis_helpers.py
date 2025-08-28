import os
import glob
import csv

import pandas as pd
import numpy as np

# from general.envelopePythonSource import get_frontiers_py

encoding = 'utf-8-sig'

def file_renamer(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

    for file_path in txt_files:
        # Open the file and read the content as CSV with semicolon delimiter
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file, delimiter=';')
            lines = list(csv_reader)  # Read all lines into a list

        # Ensure the file has at least 5 lines (0-indexed, so index 4 is the 5th line)
        if len(lines) >= 5:
            try:
                number = lines[4][1]  # Assuming index 1 for column 2
            except IndexError:
                print(f"IndexError: Skipping file {file_path} due to insufficient columns")
                continue
        else:
            print(f"Skipping file {file_path} due to insufficient lines")
            continue

        # Generate new file name
        new_file_name = f"{number.strip()}.txt"  # Strip to remove any extra whitespace
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        try:
            os.rename(file_path, new_file_path)
            print(f"Renamed {file_path} to {new_file_path}")
        except Exception as e:
            print(f"Error renaming {file_path} to {new_file_path}: {e}")


def create_all_dataframes(directory, skip_rows, type):
    all_dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            file_name, df = process_file(file_path, skip_rows, type)
            all_dataframes[file_name] = df

    return all_dataframes


def process_file(file_path, skip_rows, type):
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    signal_names = lines[0].strip().split(';')
    descriptive_names = lines[1].strip().split(';')
    units = lines[2].strip().split(';')

    if type == 'konti':
        combined_headers = [f"{desc}" for sig, desc, unit in zip(signal_names, descriptive_names, units)]
    elif type == 'trio':
        combined_headers = [f"{sig}_{desc}" for sig, desc, unit in zip(signal_names, descriptive_names, units)]
    else:
        combined_headers = [f"{sig}_{desc}_{unit}" for sig, desc, unit in zip(signal_names, descriptive_names, units)]

    df = pd.read_csv(file_path, skiprows=skip_rows, header=None, delimiter=';', encoding=encoding)

    df.columns = combined_headers[:len(df.columns)]  # Ensure column length matches

    df.set_index(df.columns[0], inplace=True)

    file_name = os.path.basename(file_path).split('.')[0]

    return file_name, df


def find_zero_crossings(gradient):
    zero_crossings = np.where(np.diff(np.sign(gradient)))[0]
    return zero_crossings


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def refine_dataframes(all_dataframes, type):
    mask_increase = 0.01

    Results = {}
    if type == 'konti':
        for k in all_dataframes.keys():
            df = all_dataframes[k]
            df['Sum Walzkraft DUO'] = df['Walzkraft  DUO-Walzwerk VL'] + df['Walzkraft  DUO-Walzwerk VR'] + df[
                'Walzkraft  DUO-Walzwerk HL'] + df['Walzkraft  DUO-Walzwerk HR']

            # Moving Averages
            all_stands_forces = ['Sum Walzkraft DUO', 'Walzkraft Walzgerüst 1', 'Walzkraft Walzgerüst 2',
                                 'Walzkraft Walzgerüst 3', 'Walzkraft Walzgerüst 4']
            for i, force in enumerate(all_stands_forces):
                if i == 0:
                    df['MAvg Walzkraft DUO'] = df[force].rolling(window=15, center=True, min_periods=1).mean()
                else:
                    df[f'MAvg Walzkraft WG{i}'] = df[force].rolling(window=15, center=True, min_periods=1).mean()

            all_stands_torques = ['Moment DUO-Walzwerk', 'Moment Walzgerüst 1', 'Moment Walzgerüst 2',
                                  'Moment Walzgerüst 3', 'Moment Walzgerüst 4']
            for i, torque in enumerate(all_stands_torques):
                if i == 0:
                    df['MAvg Moment DUO'] = df[torque].rolling(window=15, center=True, min_periods=1).mean()
                else:
                    df[f'MAvg Moment WG{i}'] = df[torque].rolling(window=15, center=True, min_periods=1).mean()

            all_temperatures = ['Temperatur 1 (Rollgang)', 'Temperatur 2 (Duo Austritt)', 'Temperatur 3 (Duo Anstich)',
                                'Temperatur 4 (vor G1)', 'Temperatur 5 (vor G2)',
                                'Temperatur 6 (vor G3)', 'Temperatur 7 (vor G4)', 'Temperatur 8 (nach G4)',
                                'Temperatur 10 (Treiber 42)']
            for i, temp in enumerate(all_temperatures):
                df[f'MAvg {temp}'] = df[temp].rolling(window=15, center=True, min_periods=1).mean()

            # Pass Verifier DUO
            force = 'MAvg Walzkraft DUO'
            df['gradient'] = np.gradient(df[force])
            bottom_mean = df[force].sort_values().head(round(0.9 * len(df[force]))).mean()
            raised_bottom_mean = bottom_mean + 15
            df['raised_bottom_mean'] = raised_bottom_mean
            for pass_number in df['aktuelle Stichnummer'].unique():
                pass_df = df[(df[force] > df['raised_bottom_mean']) & (df['aktuelle Stichnummer'] == pass_number)]
                zero_crossings = find_zero_crossings(pass_df['gradient'].values)
                if len(zero_crossings) >= 2:
                    first_zero_crossing = zero_crossings[0]
                    last_zero_crossing = zero_crossings[-1]
                    mask = (df.index >= (pass_df.index[first_zero_crossing] - mask_increase * 0.6)) & (
                                df.index <= (pass_df.index[last_zero_crossing] + mask_increase * 1.4))
                    df.loc[mask & (df['aktuelle Stichnummer'] == pass_number), 'pass_verifier_DUO'] = pass_number
                else:
                    df.loc[df['aktuelle Stichnummer'] == pass_number, 'pass_verifier_DUO'] = np.nan

            # Pass Verifier WGs
            finishing_stands_forces_MAvgs = ['MAvg Walzkraft WG1', 'MAvg Walzkraft WG2', 'MAvg Walzkraft WG3',
                                             'MAvg Walzkraft WG4']

            for i, force in enumerate(finishing_stands_forces_MAvgs):
                df['gradient'] = np.gradient(df[force])
                bottom_mean = df[force].sort_values().head(round(0.9 * len(df[force]))).mean()
                raised_bottom_mean = bottom_mean + 15
                df['raised_bottom_mean'] = raised_bottom_mean

                if df[force].max() > raised_bottom_mean:
                    pass_number = df['aktuelle Stichnummer'].max()
                    pass_df = df[df[force] > df['raised_bottom_mean']]
                    zero_crossings = find_zero_crossings(pass_df['gradient'].values)
                    if len(zero_crossings) >= 2:
                        first_zero_crossing = zero_crossings[0]
                        last_zero_crossing = zero_crossings[-1]
                        mask = (df.index >= (pass_df.index[first_zero_crossing] - mask_increase * 0.6)) & (
                                    df.index <= (pass_df.index[last_zero_crossing] + mask_increase * 1.4))
                        df.loc[mask & (df[
                                           'aktuelle Stichnummer'] == pass_number), f'pass_verifier_WG{i + 1}'] = pass_number + i
                    else:
                        df.loc[df['aktuelle Stichnummer'] == pass_number, f'pass_verifier_WG{i + 1}'] = np.nan

            drop_columns = ['gradient', 'raised_bottom_mean']
            df.drop(columns=drop_columns, inplace=True)

            # Correct and Shift Values
            # DUO
            stands = ['DUO', 'WG1', 'WG2', 'WG3', 'WG4']
            values_to_shift = ['Walzkraft', 'Moment']
            for stand in stands:
                for value in values_to_shift:
                    df['verifier_for_bottom_average'] = np.nan
                    if f'pass_verifier_{stand}' in df.columns:
                        for pass_number in df[f'pass_verifier_{stand}'].unique():
                            if pass_number >= 1:
                                first_index = df[f'pass_verifier_{stand}'].where(
                                    df[f'pass_verifier_{stand}'] == pass_number).first_valid_index()
                                last_index = df[f'pass_verifier_{stand}'].where(
                                    df[f'pass_verifier_{stand}'] == pass_number).last_valid_index()

                                first_index_x_shift1 = first_index - 1.2
                                first_index_x_shift2 = first_index - 0.2
                                last_index_x_shift1 = last_index + 0.2
                                last_index_x_shift2 = last_index + 1.2
                                df.loc[first_index_x_shift1:first_index_x_shift2, 'verifier_for_bottom_average'] = 1
                                df.loc[last_index_x_shift1:last_index_x_shift2, 'verifier_for_bottom_average'] = 1

                                df['helper'] = np.nan

                                if stand == 'DUO':
                                    df.loc[(df['verifier_for_bottom_average'] == 1) & (
                                                df['aktuelle Stichnummer'] == pass_number), 'helper'] = df[
                                        f'MAvg {value} {stand}']
                                    df.loc[df['aktuelle Stichnummer'] == pass_number, f'MAvg {value} {stand}'] = df[
                                                                                                                     f'MAvg {value} {stand}'] - \
                                                                                                                 df[
                                                                                                                     'helper'].mean()

                                else:
                                    df.loc[df['verifier_for_bottom_average'] == 1, 'helper'] = df[f'MAvg {value} {stand}']
                                    df[f'MAvg {value} {stand}'] -= df['helper'].mean()

            # Transport Temperaturen

            df['temp_gradient_DUO_Austritt'] = np.gradient(df['MAvg Temperatur 2 (Duo Austritt)']) + 1000
            df['temp_gradient_DUO_Anstich'] = np.gradient(df['MAvg Temperatur 3 (Duo Anstich)']) + 1000

            df['max_envelop_temp_duo_austritt'] = df['MAvg Temperatur 2 (Duo Austritt)'].rolling(window=1500, center=True,
                                                                                                 min_periods=1).max()

            df['mean_temp_duo_austritt'] = df['MAvg Temperatur 2 (Duo Austritt)'].rolling(window=150, center=True,
                                                                                          min_periods=1).mean()



            # Build Result Dict
            all_MAvg_forces = ['MAvg Walzkraft DUO', 'MAvg Walzkraft WG1', 'MAvg Walzkraft WG2', 'MAvg Walzkraft WG3',
                               'MAvg Walzkraft WG4']
            all_MAvg_torques = ['MAvg Moment DUO', 'MAvg Moment WG1', 'MAvg Moment WG2', 'MAvg Moment WG3',
                                'MAvg Moment WG4']
            all_rotational_frequencies = ['Reversier Duo-Walzgerüst Drehzahl-Istwert', 'Walze 1 Drehzahl-Istwert',
                                          'Walze 2 Drehzahl-Istwert', 'Walze 3 Drehzahl-Istwert',
                                          'Walze 4 Drehzahl-Istwert']
            verifiers = ['pass_verifier_DUO', 'pass_verifier_WG1', 'pass_verifier_WG2', 'pass_verifier_WG3',
                         'pass_verifier_WG4']
            pass_one = 1
            Results[k] = {}
            for i, verifier in enumerate(verifiers):
                if i == 0:
                    for pv in range(1, int(df[verifier].max()) + 1):
                        df['helper'] = np.nan
                        df.loc[df[verifier] == pv, 'helper'] = df[all_MAvg_forces[i]]
                        max_force_index = df['helper'].idxmax()
                        max_force = df['helper'].max()
                        mean_force = df['helper'].mean()

                        df['helper'] = np.nan
                        df.loc[df[verifier] == pv, 'helper'] = df[all_MAvg_torques[i]]
                        mean_torque = df['helper'].mean()

                        df['helper'] = np.nan
                        df.loc[df[verifier] == pv, 'helper'] = df[all_rotational_frequencies[i]]
                        mean_rotational_frequency = abs(df['helper'].mean()) / 60

                        Results[k][f'Pass_{pv}'] = {'max_force': max_force, 'mean_force': mean_force,
                                                    'max_force_index': max_force_index, 'mean_torque': mean_torque,
                                                    'mean_rotational_frequency': mean_rotational_frequency}
                else:
                    df['helper'] = np.nan
                    if verifier in df.columns:
                        df.loc[df[verifier] >= 1, 'helper'] = df[all_MAvg_forces[i]]
                        max_force_index = df['helper'].idxmax()
                        max_force = df['helper'].max()
                        mean_force = df['helper'].mean()

                        df['helper'] = np.nan
                        df.loc[df[verifier] >= 1, 'helper'] = df[all_MAvg_torques[i]]
                        mean_torque = df['helper'].mean()

                        df['helper'] = np.nan
                        df.loc[df[verifier] >= 1, 'helper'] = df[all_rotational_frequencies[i]]
                        mean_rotational_frequency = abs(df['helper'].mean())

                        pass_number_WG = int(df[verifiers[0]].max() + i)
                        Results[k][f'Pass_{pass_number_WG}'] = {'max_force': max_force, 'mean_force': mean_force,
                                                                'max_force_index': max_force_index,
                                                                'mean_torque': mean_torque,
                                                                'mean_rotational_frequency': mean_rotational_frequency}

            df.drop(columns='helper', inplace=True)

            for i, verifier in enumerate(verifiers):
                if i == 0:
                    for pass_number in df[verifier].unique():
                        if 1 <= pass_number < df[verifier].max():
                            try:
                                first_index = df[verifier].where(
                                    df[verifier] == pass_number).first_valid_index()
                                last_index = df[verifier].where(
                                    df[verifier] == pass_number).last_valid_index()
                                first_next_index = df[verifier].where(
                                    df[verifier] == pass_number + 1).first_valid_index()
                                Results[k][f'Pass_{int(pass_number)}'][
                                    'transport_time_after_pass'] = first_next_index - last_index
                                Results[k][f'Pass_{int(pass_number)}'][
                                    'pass_duration'] = last_index - first_index
                            except KeyError:
                                print(f'Results of pass {int(pass_number)} incomplete.')
                        elif pass_number == df[verifier].max():
                            try:
                                first_index = df[verifier].where(
                                    df[verifier] == pass_number).first_valid_index()
                                last_index = df[verifier].where(
                                    df[verifier] == pass_number).last_valid_index()
                                first_next_index = df[verifiers[1]].where(
                                    df[verifiers[1]] >= 1).first_valid_index()
                                Results[k][f'Pass_{int(pass_number)}'][
                                    'transport_time_after_pass'] = first_next_index - last_index
                                Results[k][f'Pass_{int(pass_number)}'][
                                    'pass_duration'] = last_index - first_index
                            except KeyError:
                                print(f'Results of pass {int(pass_number)} incomplete.')
                elif 0 < i < 4:
                    try:
                        Results[k][f'Pass_{int(df[verifiers[0]].max() + i)}']['transport_length_after_pass'] = 1.5
                    except KeyError:
                        print(f'cant write transport length for df {k} at step {i}')

                if 0 < i <= 4:
                    try:
                        first_index = df[verifier].first_valid_index()
                        last_index = df[verifier].last_valid_index()
                        Results[k][f'Pass_{int(df[verifiers[0]].max() + i)}'][
                            'pass_duration'] = last_index - first_index
                    except KeyError:
                        print(f'cant write pass duration for df {k} at step {i}')

            all_dataframes[k] = df.sort_index(axis=1)

        return all_dataframes, Results

    if type == 'trio':
        for k in all_dataframes.keys():
            df = all_dataframes[k]
            df['Sum_RollForce_G1'] = df['10_F3_6'] + df['11_F3_5']
            df['Sum_RollForce_G2'] = df['12_F3_4'] + df['13_F3_3']
            df['Sum_RollForce_G3'] = df['14_F3_2'] + df['15_F3_1']

            # Moving Averages
            all_stands_forces = ['Sum_RollForce_G1', 'Sum_RollForce_G2', 'Sum_RollForce_G3']
            for i, force in enumerate(all_stands_forces):
                df[f'MAvg_{force}'] = df[force].rolling(window=3, center=True, min_periods=1).mean()

            numbers = ['16', '17', '18']
            all_stands_torques = ['Mo', 'Mm', 'Mu']
            for i, torque in enumerate(all_stands_torques):
                df[f'MAvg_{torque}'] = df[f'{numbers[i]}_{torque}'].rolling(window=3, center=True, min_periods=1).mean()

            numbers = ['19', '110', '111']
            all_temperatures = ['P1', 'P2', 'P3']
            for i, temp in enumerate(all_temperatures):
                df[f'MAvg_{temp}'] = df[f'{numbers[i]}_{temp}'].rolling(window=3, center=True, min_periods=1).mean()

            # Pass Verifier DUO
            df['all_forces'] = df[['MAvg_Sum_RollForce_G1', 'MAvg_Sum_RollForce_G2', 'MAvg_Sum_RollForce_G3']].max(axis=1)
            force = 'all_forces'
            df['gradient'] = np.gradient(df[force])
            bottom_mean = df[force].sort_values().head(round(0.9 * len(df[force]))).mean()
            raised_bottom_mean = bottom_mean + 3.5
            df['raised_bottom_mean'] = raised_bottom_mean
            df['above_threshold'] = df['all_forces'] > raised_bottom_mean
            df['group'] = (df['above_threshold'] != df['above_threshold'].shift()).cumsum()
            df['group'] = df.apply(lambda x: x['group'] if x['above_threshold'] else 0, axis=1)

            df['pass_verifier_trio1'] = (df['group'] != df['group'].shift()).cumsum()
            df['pass_verifier_pure'] = df['pass_verifier_trio1']
            df.loc[~df['above_threshold'], 'pass_verifier_trio1'] = np.nan

            # For the fist 20s the threshold needs to be lower
            df['first20sG1'] = np.nan
            df.loc[df['pass_verifier_pure'] <= 13, 'first20sG1'] = df['Sum_RollForce_G1']
            bottom_mean = df['first20sG1'].sort_values().head(round(0.9 * len(df[force]))).mean()
            raised_bottom_mean = bottom_mean + 3.5
            df['raised_bottom_mean2'] = raised_bottom_mean
            df['above_threshold2'] = df['first20sG1'] > raised_bottom_mean
            df['group2'] = (df['above_threshold2'] != df['above_threshold2'].shift()).cumsum()
            df['group2'] = df.apply(lambda x: x['group2'] if x['above_threshold2'] else 0, axis=1)
            df['pass_verifier_trio2'] = (df['group2'] != df['group2'].shift()).cumsum()
            df.loc[~df['above_threshold2'], 'pass_verifier_trio2'] = np.nan

            df['pass_verifier_trio'] = df[['pass_verifier_trio1', 'pass_verifier_trio2']].max(axis=1)
            df['pass_verifier_trio'] /= 2

            for i, force in enumerate(all_stands_forces):
                df.loc[df[force] >= df['pass_verifier_trio'], f'pass_verifier_trio_G{i+1}'] = df['pass_verifier_trio']

            drop_columns = ['gradient',
                            'raised_bottom_mean',
                            'raised_bottom_mean2',
                            'above_threshold',
                            'group',
                            'group2',
                            'pass_verifier_trio1',
                            'pass_verifier_trio2',
                            'above_threshold2']
            df.drop(columns=drop_columns, inplace=True)

            for pass_number in df['pass_verifier_trio'].unique():
                if pass_number >= 1:
                    pass_number = int(pass_number)
                    first_index = df['pass_verifier_trio'].where(df['pass_verifier_trio'] == pass_number).first_valid_index()
                    last_index = df['pass_verifier_trio'].where(df['pass_verifier_trio'] == pass_number).last_valid_index()

                    if pass_number == 1:
                        first_next_index = df['pass_verifier_trio'].where(
                            df['pass_verifier_trio'] == pass_number + 1).first_valid_index()
                        endpoint = last_index + (first_next_index - last_index) * 0.55
                        df.loc[:endpoint, 'pass_number'] = pass_number
                    elif pass_number == 16:
                        last_previous_index = df['pass_verifier_trio'].where(
                            df['pass_verifier_trio'] == pass_number - 1).last_valid_index()
                        startpoint = first_index - (first_index - last_previous_index) * 0.55
                        df.loc[startpoint:, 'pass_number'] = pass_number
                    else:
                        first_next_index = df['pass_verifier_trio'].where(
                            df['pass_verifier_trio'] == pass_number + 1).first_valid_index()
                        last_previous_index = df['pass_verifier_trio'].where(
                            df['pass_verifier_trio'] == pass_number - 1).last_valid_index()
                        startpoint = first_index - (first_index - last_previous_index) * 0.55
                        endpoint = last_index + (first_next_index - last_index) * 0.55
                        df.loc[startpoint:endpoint, 'pass_number'] = pass_number

            # Correct and Shift Values
            # DUO
            stands = ['G1', 'G2', 'G3']
            values_to_shift = ['Mo', 'Mm', 'Mu', 'RollForce']
            for value in values_to_shift:
                df['verifier_for_bottom_average'] = np.nan
                for pass_number in df['pass_number'].unique():
                    if pass_number >= 1:
                        first_index = df['pass_verifier_trio'].where(
                            df['pass_verifier_trio'] == pass_number).first_valid_index()
                        last_index = df['pass_verifier_trio'].where(
                            df['pass_verifier_trio'] == pass_number).last_valid_index()

                        first_index_x_shift1 = first_index - 1.2
                        first_index_x_shift2 = first_index - 0.2
                        last_index_x_shift1 = last_index + 0.2
                        last_index_x_shift2 = last_index + 1.2
                        df.loc[first_index_x_shift1:first_index_x_shift2, 'verifier_for_bottom_average'] = 1
                        df.loc[last_index_x_shift1:last_index_x_shift2, 'verifier_for_bottom_average'] = 1

                        if value == 'RollForce':
                            for stand in stands:
                                df['helper'] = np.nan
                                df.loc[(df['verifier_for_bottom_average'] == 1) & (df['pass_number'] == pass_number), 'helper'] = df[f'MAvg_Sum_{value}_{stand}']
                                df.loc[df['pass_number'] == pass_number, f'MAvg_Sum_{value}_{stand}'] = df[f'MAvg_Sum_{value}_{stand}'] - df['helper'].mean()

                        else:
                            df['helper'] = np.nan
                            df.loc[(df['verifier_for_bottom_average'] == 1) & (df['pass_number'] == pass_number), 'helper'] = df[f'MAvg_{value}']
                            df.loc[df['pass_number'] == pass_number, f'MAvg_{value}'] = df[f'MAvg_{value}'] - df['helper'].mean()

            df['MAvg_all_Forces'] = 0.0
            df.loc[df['pass_verifier_trio_G1'] >= 1, 'MAvg_all_Forces'] = df['MAvg_Sum_RollForce_G1']
            df.loc[df['pass_verifier_trio_G2'] >= 1, 'MAvg_all_Forces'] = df['MAvg_Sum_RollForce_G2']
            df.loc[df['pass_verifier_trio_G3'] >= 1, 'MAvg_all_Forces'] = df['MAvg_Sum_RollForce_G3']

            # Build Result Dict
            verifier = 'pass_verifier_trio'
            Results[k] = {}
            for pv in range(1, int(df[verifier].max()) + 1):
                if pv > 8:
                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_all_Forces']
                    max_force = df['helper'].max()
                    mean_force = df['helper'].mean()

                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_Mo']
                    mean_torque_Mo = df['helper'].mean()

                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_Mm']
                    mean_torque_Mm = df['helper'].mean()

                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_Mu']
                    mean_torque_Mu = df['helper'].mean()

                    Results[k][f'Pass_{pv}'] = {'max_force': max_force,
                                                'mean_force': mean_force,
                                                'mean_torque_Mo': mean_torque_Mo,
                                                'mean_torque_Mm': mean_torque_Mm,
                                                'mean_torque_Mu': mean_torque_Mu}
                else:
                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_all_Forces']
                    max_force = df['helper'].max()
                    mean_force = df['helper'].sort_values().head(round(0.5 * len(df['helper']))).mean()

                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_Mo']
                    mean_torque_Mo = df['helper'].mean()

                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_Mm']
                    mean_torque_Mm = df['helper'].mean()

                    df['helper'] = np.nan
                    df.loc[df[verifier] == pv, 'helper'] = df['MAvg_Mu']
                    mean_torque_Mu = df['helper'].mean()

                    Results[k][f'Pass_{pv}'] = {'max_force': max_force,
                                                'mean_force': mean_force,
                                                'mean_torque_Mo': mean_torque_Mo,
                                                'mean_torque_Mm': mean_torque_Mm,
                                                'mean_torque_Mu': mean_torque_Mu}

            df.drop(columns='helper', inplace=True)

            # Transport Zeiten
            for pass_number in df['pass_number'].unique():
                if 1 <= pass_number < 16:
                    last_index = df['pass_verifier_trio'].where(
                        df['pass_verifier_trio'] == pass_number).last_valid_index()
                    first_next_index = df['pass_verifier_trio'].where(
                        df['pass_verifier_trio'] == pass_number + 1).first_valid_index()
                    Results[k][f'Pass_{int(pass_number)}']['Transport_time_after_pass'] = first_next_index - last_index

            all_dataframes[k] = df.sort_index(axis=1)

        return all_dataframes, Results


def transfer_Results_to_df(Results, all_dataframes, type):
    if type == 'konti':
        for k in Results.keys():
            df = all_dataframes[k]
            for stand in ['DUO', 'WG1', 'WG2', 'WG3', 'WG4']:
                df[f'mean_force_{stand}'] = np.nan
                df[f'max_force_{stand}'] = np.nan
                df[f'mean_torque_{stand}'] = np.nan
                df[f'mean_rotational_frequency_{stand}'] = np.nan
                if f'pass_verifier_{stand}' in df.columns:
                    for pass_number in df[f'pass_verifier_{stand}'].unique():
                        if pd.notna(pass_number):
                            pass_stats = Results[k][f'Pass_{int(pass_number)}']
                            df.loc[df[f'pass_verifier_{stand}'] == pass_number, f'mean_force_{stand}'] = pass_stats[
                                'mean_force']
                            df.loc[df[f'pass_verifier_{stand}'] == pass_number, f'max_force_{stand}'] = pass_stats[
                                'max_force']
                            df.loc[df[f'pass_verifier_{stand}'] == pass_number, f'mean_torque_{stand}'] = pass_stats[
                                'mean_torque']
                            df.loc[df[f'pass_verifier_{stand}'] == pass_number, f'mean_rotational_frequency_{stand}'] \
                                = pass_stats['mean_rotational_frequency']

            all_dataframes[k] = df.sort_index(axis=1)

        return all_dataframes
    elif type == 'trio':
        for k in Results.keys():
            df = all_dataframes[k]
            df[f'max_force'] = np.nan
            df[f'mean_force'] = np.nan
            df[f'mean_torque_Mo'] = np.nan
            df[f'mean_torque_Mm'] = np.nan
            df[f'mean_torque_Mu'] = np.nan
            for pass_number in df['pass_verifier_trio'].unique():
                if pd.notna(pass_number):
                    pass_stats = Results[k][f'Pass_{int(pass_number)}']
                    df.loc[df[f'pass_verifier_trio'] == pass_number, f'max_force'] = pass_stats['max_force']
                    df.loc[df[f'pass_verifier_trio'] == pass_number, f'mean_force'] = pass_stats['mean_force']
                    df.loc[df[f'pass_verifier_trio'] == pass_number, f'mean_torque_Mo'] = pass_stats['mean_torque_Mo']
                    df.loc[df[f'pass_verifier_trio'] == pass_number, f'mean_torque_Mm'] = pass_stats['mean_torque_Mm']
                    df.loc[df[f'pass_verifier_trio'] == pass_number, f'mean_torque_Mu'] = pass_stats['mean_torque_Mu']

            all_dataframes[k] = df.sort_index(axis=1)

        return all_dataframes


def drop_columns(Results, all_dataframes, type):
    if type == 'konti':
        columns_to_drop = ['Kühlwasserstrecke 1.1 Sollwert',
                           'Kühlwasserstrecke 1.1 Istwert',
                           'Kühlwasserstrecke 1.2 Sollwert',
                           'Kühlwasserstrecke 1.2 Istwert',
                           'Kühlwasserstrecke 1.3 Sollwert',
                           'Kühlwasserstrecke 1.3 Istwert',
                           'Kühlwasserstrecke 1.4 Sollwert',
                           'Kühlwasserstrecke 1.4 Istwert',
                           'Zwischenkühlsterecke 1 Sollwert',
                           'Zwischenkühlsterecke 1 Istwert',
                           'Zwischenkühlsterecke 2 Sollwert',
                           'Zwischenkühlsterecke 2 Istwer',
                           'Zwischenkühlsterecke 3 Sollwert',
                           'Zwischenkühlsterecke 3 Istwer',
                           'Kühlsterecke 2 Sollwert',
                           'Kühlsterecke 2 Istwert',
                           'Starttrigger',
                           'Induktionsofen Temp Soll',
                           'Induktionsofen Temp Ist',
                           'Auto Band Schritt 30 - Temperatur Pendeln erreicht, Anforderung RO öffnen',
                           'Reversier Duo-Walzgerüst Spannung-Istwert',
                           'Reversier Duo-Walzgerüst Ankerstrom-Istwert',
                           'Walze 1 Spannung-Istwert',
                           'Walze 1 Ankerstrom-Istwert',
                           'Walze 2 Spannung-Istwert',
                           'Walze 2 Ankerstrom-Istwert',
                           'Walze 3 Spannung-Istwert',
                           'Walze 3 Ankerstrom-Istwert',
                           'Walze 4 Spannung-Istwert',
                           'Walze 4 Ankerstrom-Istwert',
                           'Draht-Treiber 41 Drehzahl-Istwert',
                           'Draht-Treiber 41 Spannung-Istwert',
                           'Draht-Treiber 41 Ankerstrom-Istwert',
                           'Draht-Treiber 42 Drehzahl-Istwert',
                           'Draht-Treiber 42 Spannung-Istwert',
                           'Draht-Treiber 42 Ankerstrom-Istwert',
                           'N1Windungsleger Drehzahl-Istwert',
                           'N1Windungsleger Spannung-Istwert',
                           'N1Windungsleger Ankerstrom-Istwert',
                           'Gliederbandförderer Drehzahl-Istwert',
                           'Gliederbandförderer Spannung-Istwert',
                           'Gliederbandförderer Ankerstrom-Istwert',
                           'Walze 1 Zusatzstrom',
                           'Walze 2 Zusatzstrom',
                           'Walzkraft  DUO - Walzwerk VL',
                           'Walzkraft  DUO - Walzwerk VR',
                           'Walzkraft  DUO - Walzwerk HL',
                           'Walzkraft  DUO - Walzwerk HR',
                           'Walzkraft Walzgerüst 1',
                           'Walzkraft Walzgerüst 2',
                           'Walzkraft Walzgerüst 3',
                           'Walzkraft Walzgerüst 4',
                           'Moment DUO - Walzwerk',
                           'Moment Walzgerüst 1',
                           'Moment Walzgerüst 2',
                           'Moment Walzgerüst 3',
                           'Moment Walzgerüst 4',
                           'Trigger-Pyrometer vor Duo-Walze Walzguterkennung',
                           '10:29',
                           ]

        verifiers = ['pass_verifier_WG1', 'pass_verifier_WG2', 'pass_verifier_WG3', 'pass_verifier_WG4']
        torque_force = ['Moment', 'Walzkraft']
        rpm_torque = ['Drehzahl-Istwert', 'Motor-Moment Ist']
        m_averages = ['MAvg Moment ', 'MAvg Walzkraft ', 'mean_force_', 'mean_torque_', 'max_force_']

        for k in Results.keys():
            df = all_dataframes[k]

            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            for i, verifier in enumerate(verifiers):
                i += 1
                if verifier not in df.columns:
                    for p in torque_force:
                        try:
                            df.drop(columns=[f'{p} Walzgerüst {i}'], inplace=True)
                        except KeyError:
                            pass
                    for p in rpm_torque:
                        try:
                            df.drop(columns=[f'Walze {i} {p}'], inplace=True)
                        except KeyError:
                            pass
                    for p in m_averages:
                        try:
                            df.drop(columns=[f'{p}WG{i}'], inplace=True)
                        except KeyError:
                            pass

    elif type == 'trio':
        columns_to_drop = ['0_0_0_0', '0_1_0_1', '10_F3_6', '11_F3_5', '12_F3_4', '13_F3_3', '14_F3_2',
                           '15_F3_1', '16_Mo', '17_Mm', '18_Mu', '19_P1', '110_P2', '111_P3',
                           'verifier_for_bottom_average', 'pass_verifier_trio_G1', 'pass_verifier_trio_G2',
                           'pass_verifier_trio_G3', 'first20sG1', 'pass_verifier_pure', 'all_forces',
                           'Sum_RollForce_G3', 'Sum_RollForce_G2', 'Sum_RollForce_G1', '20_F_Antrieb',
                           '21_F_Abtrieb', '22_FS_Antrieb', '23_FS_Abtrieb', '24_Mo',
                           '25_Mu', '26_P1', '27_P2', '28_P3', '213_Geschwindigkeit', '214_Strom', '215_Walzspalt'
                           ]
        for k in Results.keys():
            df = all_dataframes[k]

            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return all_dataframes
