**Classificatore di generi musicali**  
Matteo Collina  
Università degli Studi di Milano    
[PAPER](https://github.com/matteocollina/get-genre/blob/master/_IRDocs/IR_MatteoCollina.pdf)

Prerequisites :
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
- [musixmatch](https://github.com/hudsonbrendon/python-musixmatch)
- numpy
- pymongo
- [youtube_dl](https://ytdl-org.github.io/youtube-dl/index.html)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- pydub
- shutil

1. Start mongo server  
`$ mongo`

2. It's possible to build ground truth only with Musixmatch service,
so create _api.json_ and put you API key `{musixmatch:<API_KEY>}`.

3. Set `__map_genres_id`: it is a hashmap that contains:  
`genre(string):id(number)`
- genre: name of genre in your local music/ folder
- id: id of genre in Musixmatch API
So the system download a list of tracks of current genre
and put them in music collection in mongodb.

4. Start to download all tracks with `createGroundTruth`,
and automatically put them in /music/genre folder.