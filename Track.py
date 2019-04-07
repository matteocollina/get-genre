# -*- coding: UTF-8 -*-
from Genre import Genre
class Track:
    def __init__(self, data):
        try:
            self.artist_name = data["artist_name"]
            self.track_name = data["track_name"]
            self.instrumental = data["instrumental"]
            self.track_length = data["track_length"]
            self.primary_genres = []
            for pg in data["primary_genres"]:
                self.primary_genres.append(Genre(pg))
            self.secondary_genres = []
            if "secondary_genres" in data:
                for sg in data["secondary_genres"]:
                    self.secondary_genres.append(Genre(sg))
        except KeyError:
            print("Error in __init__ of Track: Key not exists")
