def combine_data(raw_data, num_clients):
    """ combine test dataset.  """
    all_data = {}
    x, y = [], []
    num = 0
    for i in range(num_clients):
        x += raw_data['user_data'][i]['x']
        y += raw_data['user_data'][i]['y']
        num += raw_data['num_samples'][i]
    all_data['user_data'] = {}
    all_data['user_data']['x'] = x
    all_data['user_data']['y'] = y
    all_data['num_samples'] = num
    return all_data
