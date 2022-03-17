import torch
import numpy as np
import random
import pandas as pd
import datetime
from string import Formatter

def load_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Best precision
    try:
        bestPrecision = checkpoint['best_prec1']
        print("- Best precision loaded : {}".format(bestPrecision))
    except:
        bestPrecision = 0
        print("- No best precision present in this model")
        pass

    return model, bestPrecision

def save_checkpoint(state,is_best, savePath, epoch):
    if is_best:
        path = str(savePath) + '/best/' + 'model_best_epoch-{}.pth'.format(epoch)
    else:
        path = str(savePath)+'/checkpoint/' + 'checkpoint_epoch-{}.pth'.format(epoch)
    torch.save(state, path)

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataframe_load_test(filename):
    """
    Load the dataframe for the test set csv format of VisDrone
    @param filename: csv path
    @return: dataframe of columns [frame, x, y]
    """
    df = pd.read_csv(filename, header=None, index_col=False)
    df.columns = ['frame', 'head_id', 'x', 'y', 'width', 'height', 'out', 'occl', 'undefinied', 'undefinied']
    df['x'] = df['x'] + df['width'] // 2
    df['y'] = df['y'] + df['height'] // 2

    df = df[(df['frame'] % 10) == 1]
    df['frame'] = df['frame'] // 10 + 1
    return df[['frame', 'x', 'y']]

def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can 
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the  
    default, which is a datetime.timedelta object.  Valid inputtype strings: 
        's', 'seconds', 
        'm', 'minutes', 
        'h', 'hours', 
        'd', 'days', 
        'w', 'weeks'
    """
    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)

def estimatedTime(singleTimeEpoch, i, totalEpochs, e, tempoTrascorso):

        epochRimanenti = totalEpochs - e

        if(epochRimanenti == 0):
            return 0

        if(i == 0):
            tempoSingoloEpoch = datetime.timedelta(seconds=singleTimeEpoch)
            tempoTrascorso = tempoSingoloEpoch

            nowDate = datetime.timedelta(seconds = datetime.datetime.now().timestamp())
            remainingTime = datetime.timedelta(seconds = ((nowDate + datetime.timedelta(seconds=tempoTrascorso.total_seconds() * epochRimanenti)).total_seconds()) - nowDate.total_seconds())

            print("- Tempo trascorso in media per un singolo epoch: {}".format(str(tempoTrascorso).split(".")[0]))
            print("- Orario di completamento stimato: {:%H:%M:%S del %d %B %Y} ({})".format(datetime.datetime.now() + datetime.timedelta(seconds=tempoTrascorso.total_seconds() * epochRimanenti), strfdelta(remainingTime,"{D:02} giorni, {H:02} ore, {M:02} minuti rimanenti")))
        else:
            tempoSingoloEpoch = datetime.timedelta(seconds=singleTimeEpoch)
            tempoTrascorso = tempoTrascorso + tempoSingoloEpoch
            temp = datetime.timedelta(seconds=tempoTrascorso.total_seconds() / (i + 1))

            nowDate = datetime.timedelta(seconds = datetime.datetime.now().timestamp())

            remainingTime = datetime.timedelta(seconds = ((nowDate + datetime.timedelta(seconds=temp.total_seconds() * epochRimanenti)).total_seconds()) - nowDate.total_seconds())

            print("- Tempo trascorso in media per un singolo epoch: {}".format(str(temp).split(".")[0]))
            print("- Orario di completamento stimato: {:%H:%M:%S del %d %B %Y} ({})".format(datetime.datetime.now() + datetime.timedelta(seconds=temp.total_seconds() * epochRimanenti), strfdelta(remainingTime,"{D:02} giorni, {H:02} ore, {M:02} minuti rimanenti")))
        return tempoTrascorso
