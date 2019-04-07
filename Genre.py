# -*- coding: UTF-8 -*-
class Genre:
    def __init__(self, data):
        try:
            self.music_genre_parent_id = data["music_genre_parent_id"]
            self.music_genre_name_extended = data["music_genre_name_extended"]
            self.music_genre_name = data["music_genre_name"]
            self.music_genre_id = data["music_genre_id"]
            self.music_genre_vanity = data["music_genre_vanity"]
            self.localName = data["localName"]
        except KeyError:
            print("key not exists")
