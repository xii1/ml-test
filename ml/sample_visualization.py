import matplotlib
import matplotlib.pyplot as plt
import pandas

from utils import convert_plot_to_base64

matplotlib.use('Agg')


def process_sample_data():
    data = [[0, 1, 2, 3, 4], [0, 3, 5, 9, 11]]

    fig, axes = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.suptitle('Sample Data Visualization')

    axes[0].set_title('A sample plot')
    axes[0].plot(data[0], data[1])

    axes[1].set_title('A sample bar')
    axes[1].bar(data[0], data[1])

    img = convert_plot_to_base64(plt)
    return img


def process_csv_data():
    data_file = 'data/water_potability.csv'
    df = pandas.read_csv(data_file)

    ph_data = df['ph']
    hardness_data = df['Hardness']
    solids_data = df['Solids']
    chloramines_data = df['Chloramines']

    fig, axes = plt.subplots(2, 2, sharey=True, tight_layout=True)
    fig.suptitle('Water Potability Data Visualization')

    axes[0, 0].set_title('pH Hist.')
    axes[0, 0].hist(ph_data)

    axes[0, 1].set_title('Hardness Hist.')
    axes[0, 1].hist(hardness_data)

    axes[1, 0].set_title('Solids Hist.')
    axes[1, 0].hist(solids_data)

    axes[1, 1].set_title('Chloramines Hist.')
    axes[1, 1].hist(chloramines_data)

    img = convert_plot_to_base64(plt)
    return img
