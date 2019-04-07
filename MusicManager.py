# -*- coding: UTF-8 -*-
import os
import instantmusic
from pydub import AudioSegment
import shutil
import utils
'''
MusicManager is used to manage audio files
'''
class MusicManager:
    musicFolderName = "music"

    @staticmethod
    def downloadTrack(trackName,trackArtist):
        '''
        :param trackName: name of the track
        :param trackArtist: name of the artist
        :return: the full name of the track downloaded (e.g artist - trackname)
        '''
        trackName = trackName
        trackArtist = trackArtist
        query = "{0}-{1}".format(trackArtist,trackName)
        query_normalized = query.decode('ascii', errors='ignore').encode('ascii')
        search = instantmusic.qp(query_normalized)
        print("\nDownloading file {0} with query {1}...".format(query_normalized,search))
        audiofile = instantmusic.query_and_download_with_info(search=search, has_prompts=False, is_quiet=False)
        #audiofile = instantmusic.query_and_download(search=search, has_prompts=False, is_quiet=False)
        if (audiofile):
            print("[√] Downloaded '{0}'".format(audiofile))
            return audiofile
        else:
            print("[!] {0} not found".format(query))
            return None

    @staticmethod
    def toWav(fileName):
        sound = AudioSegment.from_mp3("{0}.mp3".format(fileName))
        sound = sound.set_channels(1)
        sound.export("{0}.wav".format(fileName), format="wav")

    @staticmethod
    def toWavTrack(trackName, trackArtist):
        MusicManager.toWav("{0} - {1}.mp3".format(trackArtist,trackName))

    @staticmethod
    def splitAudiofileWithName(path,name,format="wav",seconds=30):
        ms = seconds * 1000
        song = AudioSegment.from_file(path, format=format)
        lenSong = len(song)
        start = (lenSong/2) - (ms/2)
        stop = (lenSong / 2) + (ms/2)
        with open("{1}.{0}".format(format,name), "wb") as f:
            song[start:stop].export(f, format=format)


    @staticmethod
    def moveTrackToGenreFolder(trackPath, folderName):
        pathFolderName = "{0}/{1}".format(MusicManager.__getRootFolder(),folderName)
        if not os.path.exists(pathFolderName):
            os.makedirs(pathFolderName)
        pathFolderNameWithTrackName = "{0}/{1}".format(pathFolderName,utils.path_leaf(trackPath))
        print("Copy {0}\t\tTO\t\t{1}".format(trackPath,pathFolderNameWithTrackName))
        shutil.copy(trackPath, pathFolderNameWithTrackName)

    @staticmethod
    def removeTrackToFolder(trackPath):
        os.unlink(trackPath)

    @staticmethod
    def __getRootFolder():
        return "{0}/{1}".format(os.path.dirname(os.path.abspath(__file__)),MusicManager.musicFolderName)

    @staticmethod
    def getListGenreFolders():
        '''
        :return: paths of directories that contains min 2 tracks (required from "aT.fileClassification(...)")
        '''
        folders = []
        for fold in os.listdir(MusicManager.__getRootFolder()):
            _pathFolder = "{0}/{1}".format(MusicManager.__getRootFolder(), fold)
            if os.path.isdir(_pathFolder):
                if(len([name for name in os.listdir(_pathFolder)])>1):
                    folders.append(_pathFolder)
        return folders

    @staticmethod
    def existDataset():
        return os.path.isdir(MusicManager.__getRootFolder())