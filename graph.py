import pickle
import matplotlib.pyplot as plt
dataset = 'Sensor'
if dataset =='Bank':
    with open('pickle files/Bank_LR_ESA_zero_init.pckl', 'rb') as file:
        graph_data = pickle.load(file)
        print(graph_data)
    x_data = []
    y_data = []

    for i in range(0, 19):
        x_datas = []
        y_datas = []
        for entry in graph_data:
            if entry[0] == i:
                x_datas.append(entry[1] + 1)
                y_datas.append(entry[2])
        x_data.append(x_datas)
        y_data.append(y_datas)
    x_data = x_data[0]

    # Average the MSE per point
    new = []
    for i in range(17):
        sum = 0
        for data in y_data:
            sum += data[i]
        avg = sum / 17
        new.append(avg)
    y_data = new
    print(x_data)
    print(y_data)

    # graph to show all MSE on one page
    plt.plot(x_data, y_data)
    plt.xlabel('Number of unknown features')
    plt.ylabel('MSE')
    plt.title('MSE of all features against each other')
    plt.grid()
    plt.show()

elif dataset=='Satellite':
    with open('pickle files/SatelliteLR_ESA_zero_init.pckl', 'rb') as file:
        graph_data = pickle.load(file)
        print(graph_data)
    x_data = []
    y_data = []

    for i in range(0, 36):
        x_datas = []
        y_datas = []
        for entry in graph_data:
            if entry[0] == i:
                x_datas.append(entry[1] + 1)
                y_datas.append(entry[2])
        x_data.append(x_datas)
        y_data.append(y_datas)
    x_data = x_data[0]

    # Average the MSE per point
    new = []
    for i in range(32):
        sum = 0
        for data in y_data:
            sum += data[i]
        avg = sum / 17
        new.append(avg)
    y_data = new
    print(x_data)
    print(y_data)

    # graph to show all MSE on one page
    plt.plot(x_data, y_data)
    plt.xlabel('Number of unknown features')
    plt.ylabel('MSE')
    plt.title('MSE of all features against each other')
    plt.grid()
    plt.show()

elif dataset=='Sensor':
    with open('pickle files/Sensor_LR_ESA_zero_init.pckl', 'rb') as file:
        graph_data = pickle.load(file)
        print(graph_data)
    x_data = []
    y_data = []

    for i in range(0, 24):
        x_datas = []
        y_datas = []
        for entry in graph_data:
            if entry[0] == i:
                x_datas.append(entry[1] + 1)
                y_datas.append(entry[2])
        x_data.append(x_datas)
        y_data.append(y_datas)
    x_data = x_data[0]

    # Average the MSE per point
    new = []
    for i in range(21):
        sum = 0
        for data in y_data:
            sum += data[i]
        avg = sum / 17
        new.append(avg)
    y_data = new
    print(x_data)
    print(y_data)

    # graph to show all MSE on one page
    plt.plot(x_data, y_data)
    plt.xlabel('Number of unknown features')
    plt.ylabel('MSE')
    plt.title('MSE of all features against each other')
    plt.grid()
    plt.show()