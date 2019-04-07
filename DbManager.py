import pymongo

class DbManager:
    __host = "localhost"
    __db = "musicgenre"
    __collGenres = "genres"
    __collMusic = "music"

    @staticmethod
    def __getClient():
        return pymongo.MongoClient(host=DbManager.__host)

    @staticmethod
    def __getDb():
        return DbManager.__getClient()[DbManager.__db]

    @staticmethod
    def __getCollection(name):
        return DbManager.__getDb()[name]

    # Collections
    @staticmethod
    def getCollectionGenre():
        return DbManager.__getCollection(DbManager.__collGenres)

    @staticmethod
    def getCollectionMusic():
        return DbManager.__getCollection(DbManager.__collMusic)

    # Mapping
    @staticmethod
    def fromServerToLocalGenreName(serverName):
        '''

        :param serverName: name of genre downloaded by server with 3rd part service
        :return: genre name used locally
        '''
        blues = "blues"
        classical = "classical"
        country = "country"
        jazz = "jazz"
        pop = "pop"
        hiphop = "hiphop"
        rock = "rock"
        metal = "metal"
        reggae = "reggae"
        dance = "dance"
        d = {
            "Blues": blues,
            "Blues-Acoustic-Blues":blues,
            "Blues-Chicago-Blues": blues,
            "Blues-Classic-Blues": blues,
            "Blues-Contemporary-Blues": blues,
            "Blues-Country-Blues": blues,
            "Blues-Delta-Blues": blues,
            "Classical": classical,
            "Classical-Impressionist": classical,
            "Classical-Medieval-Era": classical,
            "Classical-Minimalism": classical,
            "Classical-Modern-Era": classical,
            "Classical-Opera": classical,
            "Classical-Orchestral": classical,
            "Classical-Renaissance": classical,
            "Classical-Romantic-Era": classical,
            "Classical-Wedding-Music": classical,
            "Classical-Art-Song": classical,
            "Classical-Brass-Woodwinds": classical,
            "Classical-Solo-Instrumental": classical,
            "Classical-Contemporary-Era": classical,
            "Classical-Oratorio": classical,
            "Classical-Cantata": classical,
            "Classical-Electronic": classical,
            "Classical-Sacred": classical,
            "Classical-Guitar": classical,
            "Classical-Piano": classical,
            "Classical-Violin": classical,
            "Classical-Cello": classical,
            "Classical-Percussion": classical,
            "Classical-Classical-Era": classical,
            "Classical-Avant-Garde": classical,
            "Classical-Baroque-Era": classical,
            "Classical-Chamber-Music": classical,
            "Classical-Chant": classical,
            "Classical-Choral": classical,
            "Classical-Classical-Crossover": classical,
            "Classical-Early-Music": classical,
            "Opera": classical,
            "Country": country,
            "Country-Alternative-Country": country,
            "Country-Americana": country,
            "Country-Bluegrass": country,
            "Country-Contemporary-Bluegrass": country,
            "Country-Contemporary-Country": country,
            "Country-Country-Gospel": country,
            "Country-Thai-Country": country,
            "Country-Honky-Tonk": country,
            "Country-Outlaw-Country": country,
            "Country-Traditional-Bluegrass": country,
            "Country-Urban-Cowboy": country,
            "Jazz": jazz,
            "Jazz-Big-Band": jazz,
            "Jazz-Bebop": jazz,
            "Jazz-Avant-Garde-Jazz": jazz,
            "Jazz-Contemporary-Jazz": jazz,
            "Jazz-Crossover-Jazz": jazz,
            "Jazz-Dixieland": jazz,
            "Jazz-Fusion": jazz,
            "Jazz-Latin-Jazz": jazz,
            "Jazz-Mainstream-Jazz": jazz,
            "Jazz-Ragtime": jazz,
            "Jazz-Smooth-Jazz": jazz,
            "Jazz-Vocal-Jazz": jazz,
            "Jazz-Hard-Bop": jazz,
            "Jazz-Trad-Jazz": jazz,
            "Jazz-Cool-Jazz": jazz,
            "Pop": pop,
            "Pop-Oldies": pop,
            "Pop-Adult-Contemporary": pop,
            "Pop-Britpop": pop,
            "Pop-Pop-Rock": pop,
            "Pop-Soft-Rock": pop,
            "Pop-Teen-Pop": pop,
            "Pop-Tribute": pop,
            "Pop-Shows": pop,
            "Pop-C-Pop": pop,
            "Pop-Cantopop-HK-Pop": pop,
            "Pop-Korean-Folk-Pop": pop,
            "Pop-Mandopop": pop,
            "Pop-Tai-Pop": pop,
            "Pop-Malaysian-Pop": pop,
            "Pop-Pinoy-Pop": pop,
            "Pop-Original-Pilipino-Music": pop,
            "Pop-Manilla-Sound": pop,
            "Pop-Indo-Pop": pop,
            "Hip-Hop-Rap": hiphop,
            "Hip-Hop-Rap-Alternative-Rap": hiphop,
            "Hip-Hop-Rap-Dirty-Sout": hiphop,
            "Hip-Hop-Rap-East-Coast-Rap": hiphop,
            "Hip-Hop-Rap-Gangsta-Rap": hiphop,
            "Hip-Hop-Rap-Hardcore-Rap": hiphop,
            "Hip-Hop-Rap-Hip-Hop": hiphop,
            "Hip-Hop-Rap-Latin-Rap": hiphop,
            "Hip-Hop-Rap-Old-School-Rap": hiphop,
            "Hip-Hop-Rap-Rap": hiphop,
            "Hip-Hop-Rap-Underground-Rap": hiphop,
            "Hip-Hop-Rap-West-Coast-Rap": hiphop,
            "Hip-Hop-Rap-UK-Hip-Hop": hiphop,
            "Hip-Hop-Rap-Chinese-Hip-Hop": hiphop,
            "Hip-Hop-Rap-Korean-Hip-Hop": hiphop,
            "Rock": rock,
            "Rock-Adult-Alternative": rock,
            "Rock-American-Trad-Rock": rock,
            "Rock-Arena-Rock": rock,
            "Rock-Blues-Rock": rock,
            "Rock-British-Invasion": rock,
            "Rock-Glam-Rock": rock,
            "Rock-Hard-Rock": rock,
            "Rock-Jam-Bands": rock,
            "Rock-Prog-Rock-Art-Rock": rock,
            "Rock-Psychedelic": rock,
            "Rock-Rock-Roll": rock,
            "Rock-Rockabilly": rock,
            "Rock-Roots-Rock": rock,
            "Rock-Singer-Songwriter": rock,
            "Rock-Southern-Rock": rock,
            "Rock-Surf": rock,
            "Rock-Tex-Mex": rock,
            "Rock-Chinese-Rock": rock,
            "Rock-Korean-Mex": rock,
            "Rock-Death-Metal-Black-Metal":metal,
            "Rock-Heavy-Metal": metal,
            "Rock-Hair-Metal": metal,
            "Reggae": reggae,
            "Reggae-Dancehall": reggae,
            "Reggae-Roots-Reggae": reggae,
            "Reggae-Dub": reggae,
            "Reggae-Ska": reggae,
            "Reggae-lovers-rock": reggae,
            "Dance":dance
        }
        if serverName in d:
            return d[serverName]
        else:
            return None




