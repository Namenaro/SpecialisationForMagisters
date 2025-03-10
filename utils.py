import json
from copy import deepcopy
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np

LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']


def get_signals_true(dataset):
    """
    Генерирует датасет без всяких повреждений чего-либо
    :param dataset: распакованный json с исходным файлом LUDB
    :return: датасет размерности 200x12
    """
    signals_mkV = []
    for patient_id, patient_data in dataset.items():
        signals_mkV.append(_get_patient_12_leads_signals(patient_data))
    return signals_mkV


def get_signals_damaged(dataset, wave_name, leads_names=None):
    """

    :param dataset:
    :param wave_name:
    :param leads_names:
    :return:
    """
    damaged_signals_12_mkV = []
    for patient_id, patient_data in dataset.items():
        damaged_12_signals_patient = damage_patient(patient_data, wave_name, leads_names)
        damaged_signals_12_mkV.append(damaged_12_signals_patient)
    return damaged_signals_12_mkV

def _get_patient_12_leads_signals(patient_data):
    leads_signals = []
    for lead_name in LEADS_NAMES:
        lead_signal = patient_data['Leads'][lead_name]['Signal']
        leads_signals.append(lead_signal)
    return leads_signals


def plot_lead_to_ax(signal_mV, ax, Y_max=None, Y_min=None, line_width=1, sample_rate=500):
    if Y_max is None:
        Y_max = max(signal_mV) + 0.1

    if Y_min is None:
        Y_min = min(signal_mV) - 0.1
    # Создаем маленькую сетку
    cell_time = 0.04  # Один миллметр по оси времени соотв. 0.04 секунды
    cell_voltage = 0.1  # один миллиметр по оси напряжения соответ. 0.1 милливольта

    x = np.arange(0, len(signal_mV), dtype=np.float32) / sample_rate
    _x_min = float(x[0])
    _x_max = float(x[-1] + 1 / sample_rate)

    x_min = math.ceil(_x_min / cell_time) * cell_time
    ax.set_xticks(np.arange(x_min, _x_max, cell_time), minor=True)

    y_min = math.ceil(Y_min / cell_voltage) * cell_voltage
    ax.set_yticks(np.arange(y_min, Y_max, cell_voltage), minor=True)

    # Создаем большую сетку
    cell_time_major = 0.2
    cell_voltage_major = 0.5

    x_min = math.ceil(_x_min / cell_time_major) * cell_time_major
    y_min = math.ceil(Y_min / cell_voltage_major) * cell_voltage_major

    ax.set_xticks(np.arange(x_min, _x_max, cell_time_major))
    ax.set_yticks(np.arange(y_min, Y_max, cell_voltage_major))

    # Включаем сетки
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='gray')

    # ограничиваем рисунок
    ax.set_xlim(_x_min, _x_max)
    ax.set_ylim(Y_min, Y_max)

    # Названия к осям
    # ax.set_xlabel("Секунды")
    ax.set_ylabel("мВ")

    # Убираем подписи осей для чистоты
    ax.set_xticklabels([])
    # ax.set_yticklabels([])

    # Устанавливаем  масштаб по осям
    aspect = cell_time / cell_voltage
    ax.set_aspect(aspect)

    ax.plot(x, signal_mV,
            linestyle='-',  # Сплошная линия
            linewidth=line_width,
            alpha=0.9
            )


def plot_before_after(signal_before_mkV, signal_after_mkV):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    signal_before_mV = mkV_to_mV(signal_before_mkV)
    signal_after_mV = mkV_to_mV(signal_after_mkV)

    Y_max = max(max(signal_after_mV), max(signal_before_mV))
    Y_min = min(min(signal_after_mV), min(signal_after_mV))

    ax1.set_title('Неповрежденный сигнал')
    signal_window_start = 1000
    signal_window_end = 2000
    plot_lead_to_ax(signal_before_mV[signal_window_start:signal_window_end], ax1, Y_max=Y_max, Y_min=Y_min)

    ax2.set_title('Поврежденный сигнал')
    plot_lead_to_ax(signal_after_mV[signal_window_start:signal_window_end], ax2, Y_max=Y_max, Y_min=Y_min)

    plt.show()


def mkV_to_mV(signal_mkV):
    signal_mV = [s / 1000 for s in signal_mkV]  # делим на 1000, т.к. хотим в мВ, а в датасете в мкВ
    return signal_mV


def get_intervals_bounds_of_wave_in_lead(patient_data, lead_name, wave_name):
    starts = []
    ends = []

    points_triplets = patient_data['Leads'][lead_name]['Delineation'][wave_name]
    for triplet in points_triplets:
        starts.append(triplet[0])
        ends.append(triplet[2])

    return starts, ends


def damage_lead_signal(starts, ends, lead_signal):
    damaged_signal = deepcopy(lead_signal)
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        for j in range(start, end + 1):
            damaged_signal[j] = 0
    return damaged_signal


def damage_patient(patient_data, wave_name, leads_names):
    if leads_names is None:
        leads_names = LEADS_NAMES

    damaged_12_signals_mkV = []
    for lead_name in leads_names:
        true_signal_mkV = patient_data['Leads'][lead_name]['Signal']
        starts, ends = get_intervals_bounds_of_wave_in_lead(patient_data, lead_name, wave_name)
        damaged_signal = damage_lead_signal(starts, ends, true_signal_mkV)
        damaged_12_signals_mkV.append(damaged_signal)
    return damaged_12_signals_mkV




def plot_example(wave_name, patient_num, lead_num):
    """
    Нарисовать пример удалкения волны для данного пациента в данном отведении
    :param wave_name: (str) имя волны одно из трех: 'p', 'qrs, 't'
    :param patient_num: (int) номер пациента в датасете: от 0 до 199
    :param lead_num: (int) номер отведения: от 0 до 11 (соответствие номера названию отведения см. в массиве LEADS_NAMES)
    :return:
    """
    path_to_dataset = Path('./LUDB/ecg_data_200.json')
    with open(path_to_dataset, 'r') as file:
        dataset = json.load(file)

        signals_true_mkV = get_signals_true(dataset)

        signals_QRS_damaged = get_signals_damaged(dataset, wave_name=wave_name)  # имя волны одно из трех: 'p', 'qrs, 't'
        lead_before_mkV = signals_true_mkV[patient_num][lead_num]
        lead_after_mkV = signals_QRS_damaged[patient_num][lead_num]

        plot_before_after(signal_before_mkV=lead_before_mkV, signal_after_mkV=lead_after_mkV)


if __name__ == "__main__":
    plot_example(wave_name='qrs', patient_num=0, lead_num=0)