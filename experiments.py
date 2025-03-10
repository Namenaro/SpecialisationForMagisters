from utils import get_signals_true, get_signals_damaged, mkV_to_mV, LEADS_NAMES
from pathlib import Path
import json


if __name__ == "__main__":
    path_to_dataset = Path('./LUDB/ecg_data_200.json')
    dataset = json.load(open(path_to_dataset, 'r'))

    # Неповрежденный сигнал всегда один и тот же, размерность 200 на 12
    true_mkv = get_signals_true()

    # ПРИМЕР поврежения одного отведения:
    # Повредим у всех пациентов отведение i, удалив в нем qrs
    damaged_i_qrs_mkv = get_signals_damaged(dataset, wave_name='qrs', leads_names=['i'])

    # ПРРИМЕР повреждения всех отведений:
    # Уадалим волну p во всех отведениях
    damaged_i_qrs_mkv = get_signals_damaged(dataset, wave_name='p', leads_names=LEADS_NAMES)

    # Если требуется сигнал в милливольтах, то с каждым
    # отведением каждого пациента надо сделать:
    # signal_mV = mkV_to_mV(signal_mkV)
