from dreamview_carla.dreamview import Connection
import carla
import time

# main
if __name__ == '__main__':

    # connect dreamview
    connection = Connection(None, port="8899")

    modules = ['Prediction', 'Control', 'Planning', 'Localization']

    for m in modules:
        connection.enable_module(m)

    # wait for 3s
    time.sleep(3)

    # while True:
    time.sleep(1)
    result = connection.get_module_status()
    print(result)


    connection.disconnect()