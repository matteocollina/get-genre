# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from DbManager  import DbManager
from MusicManager  import MusicManager
from AudioClassifierManager import AudioClassifierManager
from Track import Track
import utils
import os
from musixmatch import Musixmatch
from pyAudioAnalysis import audioTrainTest as aT

__projectPath = os.path.dirname(os.path.abspath(__file__))
__musixmatchAPIKey = utils.loadAPI()['musixmatch']
__map_genres_id = {"rock":21,"reggae":24,"hiphop":18,"pop":14,"jazz":11,"country":6,"classical":5,"blues":2,"metal":1153,"dance":17 }
__list_of_dirs = [f for f in MusicManager.getListGenreFolders()]
__cacheDatasetCreate = dict()

def initDbCollections(fromApi = 'Musixmatch'):
    '''
    Initialize MongoDb collection with genres and tracks foreach genre

    :param fromApi: choose the APIs from which to retrieve the genres (available: MusixMatch)
    :return:
    '''
    if(fromApi == 'Musixmatch'):
        client = Musixmatch(__musixmatchAPIKey)
        # Genres
        if DbManager.getCollectionGenre().count()==0:
            print("Init data genres...\n")
            count_add = 0
            list_genres = client.genres_get()["message"]["body"]["music_genre_list"]
            for g in list_genres:
                count_add += 1
                genre = g["music_genre"]
                genre["localName"] = DbManager.fromServerToLocalGenreName(genre["music_genre_vanity"])
                DbManager.getCollectionGenre().insert_one(genre)
            print("√ Added {0} genre/s".format(count_add))
        else:
            print("- No genres added, {0} on database".format(DbManager.getCollectionGenre().count()))

        # Music
        if DbManager.getCollectionMusic().count() == 0:
            print("Init data music...")
            count_add = 0
            if len(__map_genres_id.items()) == 0:
                print("You have to a hashmap {genre:id,..}")
            for keyGen,valGen in __map_genres_id.items():
                count_item = 0
                for page_count in range(1,3):
                    '''
                    list_tracks = client.chart_tracks_get(f_has_lyrics=False, page=page_count, page_size=100, country=country)["message"]["body"][
                        "track_list"]
                    '''
                    list_tracks = client.track_by_genre_id(page_size=100, page=page_count,f_music_genre_id=valGen)["message"]["body"]["track_list"]

                    for t in list_tracks:
                        current = t["track"]

                        primary_genres = list()
                        exist = {}
                        exist_genre_locally = False
                        for pg in current["primary_genres"]["music_genre_list"]:
                            music_genre = pg["music_genre"]
                            music_genre_local = DbManager.fromServerToLocalGenreName(music_genre["music_genre_vanity"])
                            if DbManager.fromServerToLocalGenreName(music_genre["music_genre_vanity"]) \
                                    and (music_genre_local not in exist):
                                music_genre['localName'] = music_genre_local
                                primary_genres.append(music_genre)
                                exist[music_genre_local] = True
                                exist_genre_locally = True

                        # Add track to mongoDb only if exist
                        if exist_genre_locally:
                            count_add += 1
                            count_item += 1
                            DbManager.getCollectionMusic().insert_one({
                                "artist_name":current["artist_name"],"track_name":current["track_name"],
                                "primary_genres":primary_genres,
                                "instrumental":current["instrumental"],
                                "track_length":current["track_length"] if "track_length" in current else 0
                            })
                print("√ Added {0} track/s for genre {1}".format(count_item,keyGen))

            if count_add > 0:
                print("√ Added {0} track/s".format(count_add))
        else:
            print("- No music added, {0} on database".format(DbManager.getCollectionMusic().count()))

    else:
        print("This API is not available\n")
        return 0


def createGroundTruth():
    '''
    Download tracks stored in mongo db collection and download in the right directory (of genre)

    :return:
    '''
    list_tracks = DbManager.getCollectionMusic().find()
    list_tracks = list_tracks[utils.loadVersionOnCache():]
    for track_data in list_tracks:
        try:
            track = Track(track_data)
            if len(track.primary_genres)==0 :
                print("\n[!] {0}-{1} hasn't primary genre".format(track.artist_name,track.track_name))
            else:
                    isValidTrackName = "".join(e for e in track.track_name.encode('utf8') if e.isalnum()).isalnum()
                    # If track contains genres
                    if (len(track.primary_genres) > 0 and isValidTrackName):
                        # Download tracks
                        track_title = MusicManager.downloadTrack(track.track_name.encode('utf8'),
                                                                 track.artist_name.encode('utf8'))

                        # split audio file, export to wav
                        track_path = utils.get_pathfile_for_name_in_current_dir(__projectPath, track_title, "mp3")
                        MusicManager.splitAudiofileWithName("{0}".format(track_path), track_title,"mp3")
                        MusicManager.toWav(track_title)

                        assert os.path.exists(track_path)

                        # if it has more genres, move to each genre dir
                        for primary_genre in track.primary_genres:
                            MusicManager.moveTrackToGenreFolder(track_path, primary_genre.localName)
                        # Remove audio file in root dir
                        utils.remove_audio_files_on_current_dir()
                    utils.incrementVersionOnCache()
                    # Reset Cache for retry download
                    __cacheDatasetCreate.clear()
        except Exception as ex:
            print("EXCEPTION", ex)
            if(track.track_name not in __cacheDatasetCreate):
                __cacheDatasetCreate[track.track_name] = True
                print("RETRY another time.... ")
                createGroundTruth()
            else:
                print("SKIPPING Track {0}".format(track.track_name))



if __name__ == '__main__':

    initDbCollections()

    if not MusicManager.existDataset():
        createGroundTruth()
    else:
        print("Directoy {0} exists so system doesn't download any song".format(MusicManager.musicFolderName))


    for path in __list_of_dirs:
        if utils.items_in_dir(path)-1 <= 1:
            print("ERROR in folder {0}: Each genre folder must contains more then 1 item".format(path))
            exit(1)


    # STEP A: Feature Extraction
    [features, classNames, _] = AudioClassifierManager.getFeaturesAndClasses(__list_of_dirs)
    count_classes = AudioClassifierManager.getCountClasses(features)

    if count_classes == 0:
        print("ERROR: No classes/genres found!")
        exit(1)
    else:
        print("{0} classes found".format(count_classes))

    #Training foreach model and perTrain param
    for model in AudioClassifierManager.getAllModels():
        for pT in AudioClassifierManager.getPerTrainProportions():
            model_name = AudioClassifierManager.getModelNameForTypeAndPt(model,pT)
            print("START Train model of type {0} with perTrain param {1}".format(model,pT))

            for i, f in enumerate(features):
                print("Class {0} has {1} audio files".format(i+1,AudioClassifierManager.getCountTracksInClass(f)))
                if AudioClassifierManager.getCountTracksInClass(f) == 0:
                    print("ERROR: " + __list_of_dirs[i] + " folder is empty or non-existing!")
                    exit(1)

            # Store ARFF file, necessary for knn model
            AudioClassifierManager.writeTrainDataToARFF(model_name, features, classNames)

            # STEP B: Classifier Evaluation and Parameter Selection
            classifier_par = AudioClassifierManager.getListParamsForClassifierType(model)

            # Feature optimization:
            features = AudioClassifierManager.getFeaturesOptimized(features)

            [bestParam, result_matrix, precision_classes_all, recall_classes_all, f1_classes_all, f1_all, ac_all] = \
                    AudioClassifierManager.getResultMatrixAndBestParam\
                        (features,classNames,model,AudioClassifierManager.BEST_ACCURACY,perTrain=pT)

            print("Selected params: {0:.5f}".format(bestParam))

            AudioClassifierManager.saveConfusionMatrix(result_matrix,classNames,model_name)
            AudioClassifierManager.saveParamsFromClassification(classNames,
                                                                AudioClassifierManager.getListParamsForClassifierType(model),
                                                                model_name,precision_classes_all,recall_classes_all,
                                                                f1_classes_all,ac_all,f1_all)

            # Feature normalization:
            (features_norm, MEAN, STD) = aT.normalizeFeatures(features)

            MEAN = MEAN.tolist()
            STD = STD.tolist()
            featuresNew = features_norm

            # Re-apply classification with normalized features and best param
            finalClassifier = AudioClassifierManager.getTrainClassifier(featuresNew,model,bestParam)
            # Save final model
            AudioClassifierManager.saveClassifierModel(featuresNew,model_name,model,finalClassifier,MEAN,STD,classNames,bestParam)

