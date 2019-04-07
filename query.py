from AudioClassifierManager import AudioClassifierManager
from MusicManager import MusicManager
import pyAudioAnalysis.audioTrainTest as aT
import utils
import os
import numpy as np
import glob

if __name__ == '__main__':
    retry = 0
    track_artist = raw_input("Enter track artist/s (if more than one separated by commas): ")
    track_title = raw_input("Enter track title: ")

    '''
    Blues: Gary Coleman - The Sky is Crying
    Classical: Chopin - Nocturne op.9 No.2
    Country: Alan Jackson - Country Boy
    Dance: David Guetta - Like I Do
    Hip Hop: 50 Cents - Candy Shop
    Jazz: Ray Charles-Hit the road jack
    Metal: Semblant - What Lies Ahead 
    Pop: Bob Sinclair-I Believe
    Reggae: Alborosie - Kingston Town
    Rock: Red hot chili peppers-By the Way
    '''

    # Download file audio
    try:
        track_downloaded_name = MusicManager.downloadTrack(track_title.encode('utf8'),
                                                           track_artist.encode('utf8'))
    except Exception as e:
        print("ERROR {0}".format(e))
        if(retry < 1):
            retry += 1
            print("RETRY...")
            try:
                track_downloaded_name = MusicManager.downloadTrack(track_title.encode('utf8'),
                                                                   track_artist.encode('utf8'))
            except Exception as ex:
                print("ERROR {0}".format(ex))
                exit(1)

    filename_complete = ""
    if(track_downloaded_name):
        file_path = glob.glob('./{0}.*'.format(track_downloaded_name))[0]
        filename, file_extension = os.path.splitext(file_path)
        filename_complete = "{0}{1}".format(track_downloaded_name,file_extension)
        # Split 1 minute of song
        MusicManager.splitAudiofileWithName("{0}".format(file_path),filename,"".join(file_extension[1:]), 60)
    else:
        print("File not found, retry with another track")

    # Foreach model print probabilities ordered.
    if(filename_complete):
        for model in AudioClassifierManager.getAllModels():
            for pT in AudioClassifierManager.getPerTrainProportions():
                model_name = AudioClassifierManager.getModelNameForTypeAndPt(model, pT)
                results = dict()
                if(os.path.isfile('./{0}'.format(model_name))):
                    print("\nModel: {0}".format(model_name))
                    # get model saved
                    _fileClass = aT.fileClassification(filename_complete,model_name,model)
                    _valuePositiveInMatrix = np.concatenate( np.argwhere(_fileClass[1]>0), axis=0 )
                    for indexGen,gen in enumerate(_fileClass[2]):
                        results[gen] = _fileClass[1][indexGen]

                    for key, value in sorted(results.items(), key=lambda (k, v): (v, k),reverse=True):
                        print "{0}: {1}".format(key, format(value*100, '.2f'))
        utils.remove_audio_files_on_current_dir()
