from videoanalizer.openface.output_extractor import VidDataExtractor



def extract_samples(dx:VidDataExtractor, config:dict):
    if not all(setting in config for setting in ()):
        raise Exception("configs must contain 'interval', 'frames_per_sample','overlap' and 'only_success")
    intervals           = config['interval']
    fps                 = config['frames_per_sample']
    overlap             = config['overlap']
    only_success        = config['only_success']

    if (fps<0):
        fps = 1e10
    
    if (only_success):
        dx = dx.get_only_success()   # drop frames that can contain errors
    
    # select only the part of the video you need
    dx = dx.get_frames_in_intervals(intervals)

    # holes in the analysis
    frames = dx.csv['frame'].tolist()
    jumps = []
    for i in range(len(frames)-1):
        if frames[i+1]-frames[i] > 30:
            jumps.append(i)

    # get points in the analysys (where a jump begins)
    n_samples = int(len(frames)//fps + (len(frames)%fps==0 and 0 or 1))
    points = [x for x in range(n_samples)]
    for i,jump in enumerate(reversed(jumps)):
        points[-(i+1)] = jump+1
    
    # get points in the analysys (every fps frames)
    j=0
    for i in range(0,n_samples):
        end = (i+1)*fps
        if (end>=len(frames) or frames[i*fps] + fps +50 >= frames[end]):
            points[j] = i*fps
            j += 1

    # add some overlapping points
    if overlap:
        init = fps // (overlap+1)
        for i in range(init, len(frames), step=fps/overlap):
            points.append(i)

    # foreach point p found, get a dx in starting from p and ending in p+fps
    samples_dx = []
    for i in points:
        tmp = dx.get_frames_in_intervals([(i, i+fps)])
        samples_dx.append(tmp)

    return samples_dx
    
