# Error analysis (base Qwen, locked test format)

Locked format: `json`

Categories: correct exact-match, wrong intent, wrong slots (intent OK), malformed/unparsable.

## SNIPS (format=`json`, N=700)

Counts: {'correct': 0, 'wrong_intent': 347, 'wrong_slot': 241, 'malformed': 112}

### Correct (~5)

_None in this category._

### Wrong Intent (~5)

- **id**: `snips-test-1`
  - request: i want to bring four people to a place that s close to downtown that serves churrascaria cuisine
  - gold: `{"intent": "BookRestaurant", "slots": [{"name": "party_size_number", "value": "four"}, {"name": "spatial_relation", "value": "close"}, {"name": "poi", "value": "downtown"}, {"name": "restaurant_type", "value": "churrascaria"}]}`
  - parsed: `{"intent": "SearchCreativeWork", "slots": [{"name": "location_name", "value": "downtown"}, {"name": "restaurant_type", "value": "churrascaria"}]}`
  - raw: `{
  "intent": "SearchCreativeWork",
  "slots": [
    {
      "name": "location_name",
      "value": "downtown"
    },
    {
      "name": "restaurant_type",
      "value": "churrascaria"
    }
  ]
}`

- **id**: `snips-test-4`
  - request: play signe anderson chant music that is newest
  - gold: `{"intent": "PlayMusic", "slots": [{"name": "artist", "value": "signe anderson"}, {"name": "music_item", "value": "chant"}, {"name": "sort", "value": "newest"}]}`
  - parsed: `{"intent": "Play Music", "slots": [{"name": "song_name", "value": "signe anderson chant"}]}`
  - raw: `{
  "intent": "Play Music",
  "slots": [
    {
      "name": "song_name",
      "value": "signe anderson chant"
    }
  ]
}`

- **id**: `snips-test-5`
  - request: can you let me know what animated movies are playing close by
  - gold: `{"intent": "SearchScreeningEvent", "slots": [{"name": "movie_type", "value": "animated movies"}, {"name": "spatial_relation", "value": "close by"}]}`
  - parsed: `{"intent": "SearchCreativeWork", "slots": [{"name": "movie_name", "value": "animated_movies"}]}`
  - raw: `{
  "intent": "SearchCreativeWork",
  "slots": [
    {
      "name": "movie_name",
      "value": "animated_movies"
    }
  ]
}`

- **id**: `snips-test-6`
  - request: can you get me reservations for a highly rated restaurant in seychelles
  - gold: `{"intent": "BookRestaurant", "slots": [{"name": "sort", "value": "highly rated"}, {"name": "restaurant_type", "value": "restaurant"}, {"name": "country", "value": "seychelles"}]}`
  - parsed: `{"intent": "GetReservation", "slots": [{"name": "restaurant_name", "value": "seychelles"}, {"name": "rating_unit", "value": "highly"}]}`
  - raw: `{
  "intent": "GetReservation",
  "slots": [
    {
      "name": "restaurant_name",
      "value": "seychelles"
    },
    {
      "name": "rating_unit",
      "value": "highly"
    }
  ]
}`

- **id**: `snips-test-8`
  - request: find worldly goods starting now at a movie house
  - gold: `{"intent": "SearchScreeningEvent", "slots": [{"name": "movie_name", "value": "worldly goods"}, {"name": "timeRange", "value": "now"}, {"name": "object_location_type", "value": "movie house"}]}`
  - parsed: `{"intent": "FindWorldlyGoods", "slots": [{"name": "start_time", "value": "now"}, {"name": "movie_house", "value": "movie house"}]}`
  - raw: `{
  "intent": "FindWorldlyGoods",
  "slots": [
    {
      "name": "start_time",
      "value": "now"
    },
    {
      "name": "movie_house",
      "value": "movie house"
    }
  ]
}`

### Wrong Slot (~5)

- **id**: `snips-test-0`
  - request: add sabrina salerno to the grime instrumentals playlist
  - gold: `{"intent": "AddToPlaylist", "slots": [{"name": "artist", "value": "sabrina salerno"}, {"name": "playlist", "value": "grime instrumentals"}]}`
  - parsed: `{"intent": "AddToPlaylist", "slots": [{"name": "playlist", "value": "grime instrumentals"}, {"name": "artist", "value": "Sabrina Salerno"}]}`
  - raw: `{
  "intent": "AddToPlaylist",
  "slots": [
    {
      "name": "playlist",
      "value": "grime instrumentals"
    },
    {
      "name": "artist",
      "value": "Sabrina Salerno"
    }
  ]
}`

- **id**: `snips-test-2`
  - request: put lindsey cardinale into my hillary clinton s women s history month playlist
  - gold: `{"intent": "AddToPlaylist", "slots": [{"name": "artist", "value": "lindsey cardinale"}, {"name": "playlist_owner", "value": "my"}, {"name": "playlist", "value": "hillary clinton s women s history month playlist"}]}`
  - parsed: `{"intent": "AddToPlaylist", "slots": [{"name": "playlist", "value": "Lindsey Cardinale, Hillary Clinton"}, {"name": "artist", "value": "Hillary Clinton"}]}`
  - raw: `{
  "intent": "AddToPlaylist",
  "slots": [
    {
      "name": "playlist",
      "value": "Lindsey Cardinale, Hillary Clinton"
    },
    {
      "name": "artist",
      "value": "Hillary Clinton"
    }
  ]
}`

- **id**: `snips-test-7`
  - request: what s the weather here on 2/7/2021
  - gold: `{"intent": "GetWeather", "slots": [{"name": "current_location", "value": "here"}, {"name": "timeRange", "value": "2/7/2021"}]}`
  - parsed: `{"intent": "GetWeather", "slots": [{"name": "city", "value": "New York"}, {"name": "date", "value": "2/7/2021"}]}`
  - raw: `{
  "intent": "GetWeather",
  "slots": [
    {
      "name": "city",
      "value": "New York"
    },
    {
      "name": "date",
      "value": "2/7/2021"
    }
  ]
}`

- **id**: `snips-test-10`
  - request: what movies are playing at mann theatres
  - gold: `{"intent": "SearchScreeningEvent", "slots": [{"name": "movie_type", "value": "movies"}, {"name": "location_name", "value": "mann theatres"}]}`
  - parsed: `{"intent": "SearchScreeningEvent", "slots": [{"name": "movie_name", "value": "manne theatres"}]}`
  - raw: `{
  "intent": "SearchScreeningEvent",
  "slots": [
    {
      "name": "movie_name",
      "value": "manne theatres"
    }
  ]
}`

- **id**: `snips-test-14`
  - request: where can i purchase the tv show time for heroes
  - gold: `{"intent": "SearchCreativeWork", "slots": [{"name": "object_type", "value": "tv show"}, {"name": "object_name", "value": "time for heroes"}]}`
  - parsed: `{"intent": "SearchCreativeWork", "slots": [{"name": "tv_show_name", "value": "Heroes"}]}`
  - raw: `{
  "intent": "SearchCreativeWork",
  "slots": [
    {
      "name": "tv_show_name",
      "value": "Heroes"
    }
  ]
}`

### Malformed (~5)

- **id**: `snips-test-3`
  - request: will it snow in mt on june 13 2038
  - gold: `{"intent": "GetWeather", "slots": [{"name": "condition_description", "value": "snow"}, {"name": "state", "value": "mt"}, {"name": "timeRange", "value": "june 13 2038"}]}`
  - parsed: `None`
  - raw: `{
  "intent": "AddToPlaylist",
  "slots": [
    {
      "name": "album",
      "value": "Snow White"
    },
    {
      "name": "artist",
      "value": "The Snow White"
    },
    {
      "name": "best_rating",
      "value": "5"
    },
    {
      "name": "city",
      "value": "Mountain View"
    },
    {
      "name": "condition_description",
      "value": "Sunny"
    },
    {
      "name": "…`

- **id**: `snips-test-17`
  - request: make me a reservation in south carolina
  - gold: `{"intent": "BookRestaurant", "slots": [{"name": "state", "value": "south carolina"}]}`
  - parsed: `None`
  - raw: `{"intent":"Make a Reservation","slots":{"location_name":"South Carolina","service":"Restaurant"}}`

- **id**: `snips-test-20`
  - request: rate the current novel four of 6 stars
  - gold: `{"intent": "RateBook", "slots": [{"name": "object_select", "value": "current"}, {"name": "object_type", "value": "novel"}, {"name": "rating_value", "value": "four"}, {"name": "best_rating", "value": "6"}, {"name": "rating_unit", "value": "stars"}]}`
  - parsed: `None`
  - raw: `{"intent":"RateItem","slots":{"rating_value":"4","rating_unit":"Stars"}}`

- **id**: `snips-test-21`
  - request: add the song don t drink the water to my playlist
  - gold: `{"intent": "AddToPlaylist", "slots": [{"name": "music_item", "value": "song"}, {"name": "playlist", "value": "don t drink the water"}, {"name": "playlist_owner", "value": "my"}]}`
  - parsed: `None`
  - raw: `{"intent":"AddToPlaylist","slots":{"song":"don't drink the water"}}`

- **id**: `snips-test-23`
  - request: show the movie times
  - gold: `{"intent": "SearchScreeningEvent", "slots": [{"name": "object_type", "value": "movie times"}]}`
  - parsed: `None`
  - raw: `{"intent":"ShowMovieTimes","slots":[],"entities":[],"properties":[]}`

### Failure pattern notes

- Malformed / unparsable: 112 (16.0%)
- Wrong intent (parsable): 347
- Wrong slots only: 241
- Correct exact match: 0


## ATIS (format=`json`, N=893)

Counts: {'correct': 2, 'wrong_intent': 334, 'wrong_slot': 388, 'malformed': 169}

### Correct (~5)

- **id**: `atis-test-374`
  - request: what airline is aa
  - gold: `{"intent": "atis_airline", "slots": [{"name": "airline_code", "value": "aa"}]}`
  - parsed: `{"intent": "atis_airline", "slots": [{"name": "airline_code", "value": "aa"}]}`
  - raw: `{
  "intent": "atis_airline",
  "slots": [
    {
      "name": "airline_code",
      "value": "aa"
    }
  ]
}`

- **id**: `atis-test-628`
  - request: list the airports in la
  - gold: `{"intent": "atis_airport", "slots": [{"name": "city_name", "value": "la"}]}`
  - parsed: `{"intent": "atis_airport", "slots": [{"name": "city_name", "value": "la"}]}`
  - raw: `{
  "intent": "atis_airport",
  "slots": [
    {
      "name": "city_name",
      "value": "la"
    }
  ]
}`

### Wrong Intent (~5)

- **id**: `atis-test-0`
  - request: i would like to find a flight from charlotte to las vegas that makes a stop in st. louis
  - gold: `{"intent": "atis_flight", "slots": [{"name": "fromloc.city_name", "value": "charlotte"}, {"name": "toloc.city_name", "value": "las vegas"}, {"name": "stoploc.city_name", "value": "st. louis"}]}`
  - parsed: `{"intent": "find_flight", "slots": [{"name": "fromloc", "value": "Charlotte"}, {"name": "tolloc", "value": "St. Louis"}]}`
  - raw: `{
  "intent": "find_flight",
  "slots": [
    {
      "name": "fromloc",
      "value": "Charlotte"
    },
    {
      "name": "tolloc",
      "value": "St. Louis"
    }
  ]
}`

- **id**: `atis-test-1`
  - request: on april first i need a ticket from tacoma to san jose departing before 7 am
  - gold: `{"intent": "atis_airfare", "slots": [{"name": "depart_date.month_name", "value": "april"}, {"name": "depart_date.day_number", "value": "first"}, {"name": "fromloc.city_name", "value": "tacoma"}, {"name": "toloc.city_name", "value": "san jose"}, {"name": "depart_time.time_relative", "value": "before"}, {"name": "depart_time.time", "value": "7 am"}]}`
  - parsed: `{"intent": "atis_flight", "slots": [{"name": "departure_date", "value": "April 1st"}, {"name": "arrival_date", "value": "April 2nd"}, {"name": "flight_number", "value": "TAC-043"}]}`
  - raw: `{
  "intent": "atis_flight",
  "slots": [
    {
      "name": "departure_date",
      "value": "April 1st"
    },
    {
      "name": "arrival_date",
      "value": "April 2nd"
    },
    {
      "name": "flight_number",
      "value": "TAC-043"
    }
  ]
}`

- **id**: `atis-test-10`
  - request: find a flight from memphis to tacoma dinner
  - gold: `{"intent": "atis_flight", "slots": [{"name": "fromloc.city_name", "value": "memphis"}, {"name": "toloc.city_name", "value": "tacoma"}, {"name": "meal_description", "value": "dinner"}]}`
  - parsed: `{"intent": "find_flight", "slots": [{"name": "fromloc", "value": "Memphis"}, {"name": "tolloc", "value": "Tacoma"}]}`
  - raw: `{
  "intent": "find_flight",
  "slots": [
    {
      "name": "fromloc",
      "value": "Memphis"
    },
    {
      "name": "tolloc",
      "value": "Tacoma"
    }
  ]
}`

- **id**: `atis-test-11`
  - request: on next wednesday flight from kansas city to chicago should arrive in chicago around 7 pm return flight on thursday
  - gold: `{"intent": "atis_flight", "slots": [{"name": "depart_date.date_relative", "value": "next"}, {"name": "depart_date.day_name", "value": "wednesday"}, {"name": "fromloc.city_name", "value": "kansas city"}, {"name": "toloc.city_name", "value": "chicago"}, {"name": "toloc.city_name", "value": "chicago"}, {"name": "arrive_time.time_relative", "value": "around"}, {"name": "arrive_time.time", "value": "7 pm"}, {"name": "return_date.day_name", "value": "thursday"}]}`
  - parsed: `{"intent": "flight", "slots": [{"name": "flight_number", "value": "KANS01"}, {"name": "departure_date", "value": "2023-03-15"}, {"name": "arrival_date", "value": "2023-03-16"}]}`
  - raw: `{
  "intent": "flight",
  "slots": [
    {
      "name": "flight_number",
      "value": "KANS01"
    },
    {
      "name": "departure_date",
      "value": "2023-03-15"
    },
    {
      "name": "arrival_date",
      "value": "2023-03-16"
    }
  ]
}`

- **id**: `atis-test-12`
  - request: show flight and prices kansas city to chicago on next wednesday arriving in chicago by 7 pm
  - gold: `{"intent": "atis_flight#atis_airfare", "slots": [{"name": "fromloc.city_name", "value": "kansas city"}, {"name": "toloc.city_name", "value": "chicago"}, {"name": "depart_date.date_relative", "value": "next"}, {"name": "depart_date.day_name", "value": "wednesday"}, {"name": "toloc.city_name", "value": "chicago"}, {"name": "arrive_time.time_relative", "value": "by"}, {"name": "arrive_time.time", "value": "7 pm"}]}`
  - parsed: `{"intent": "show flight", "slots": [{"name": "flight", "value": "kansas city to chicago"}, {"name": "departure_date", "value": "next wednesday"}, {"name": "arrival_time", "value": "7 pm"}]}`
  - raw: `{
  "intent": "show flight",
  "slots": [
    {
      "name": "flight",
      "value": "kansas city to chicago"
    },
    {
      "name": "departure_date",
      "value": "next wednesday"
    },
    {
      "name": "arrival_time",
      "value": "7 pm"
    }
  ]
}`

### Wrong Slot (~5)

- **id**: `atis-test-2`
  - request: on april first i need a flight going from phoenix to san diego
  - gold: `{"intent": "atis_flight", "slots": [{"name": "depart_date.month_name", "value": "april"}, {"name": "depart_date.day_number", "value": "first"}, {"name": "fromloc.city_name", "value": "phoenix"}, {"name": "toloc.city_name", "value": "san diego"}]}`
  - parsed: `{"intent": "atis_flight", "slots": [{"name": "flight_number", "value": "Phoenix to San Diego"}, {"name": "departure_date", "value": "April 1st"}]}`
  - raw: `{
  "intent": "atis_flight",
  "slots": [
    {
      "name": "flight_number",
      "value": "Phoenix to San Diego"
    },
    {
      "name": "departure_date",
      "value": "April 1st"
    }
  ]
}`

- **id**: `atis-test-3`
  - request: i would like a flight traveling one way from phoenix to san diego on april first
  - gold: `{"intent": "atis_flight", "slots": [{"name": "round_trip", "value": "one way"}, {"name": "fromloc.city_name", "value": "phoenix"}, {"name": "toloc.city_name", "value": "san diego"}, {"name": "depart_date.month_name", "value": "april"}, {"name": "depart_date.day_number", "value": "first"}]}`
  - parsed: `{"intent": "atis_flight", "slots": [{"name": "flight_number", "value": "Phoenix-San Diego"}, {"name": "departure_date", "value": "April 1st"}]}`
  - raw: `{
  "intent": "atis_flight",
  "slots": [
    {
      "name": "flight_number",
      "value": "Phoenix-San Diego"
    },
    {
      "name": "departure_date",
      "value": "April 1st"
    }
  ]
}`

- **id**: `atis-test-4`
  - request: i would like a flight from orlando to salt lake city for april first on delta airlines
  - gold: `{"intent": "atis_flight", "slots": [{"name": "fromloc.city_name", "value": "orlando"}, {"name": "toloc.city_name", "value": "salt lake city"}, {"name": "depart_date.month_name", "value": "april"}, {"name": "depart_date.day_number", "value": "first"}, {"name": "airline_name", "value": "delta airlines"}]}`
  - parsed: `{"intent": "atis_flight", "slots": [{"name": "fromloc", "value": "Orlando"}, {"name": "to", "value": "Salt Lake City"}, {"name": "flight_date", "value": "April 1st"}]}`
  - raw: `{
  "intent": "atis_flight",
  "slots": [
    {
      "name": "fromloc",
      "value": "Orlando"
    },
    {
      "name": "to",
      "value": "Salt Lake City"
    },
    {
      "name": "flight_date",
      "value": "April 1st"
    }
  ]
}`

- **id**: `atis-test-5`
  - request: i need a flight from toronto to newark one way leaving wednesday evening or thursday morning
  - gold: `{"intent": "atis_flight", "slots": [{"name": "fromloc.city_name", "value": "toronto"}, {"name": "toloc.city_name", "value": "newark"}, {"name": "round_trip", "value": "one way"}, {"name": "depart_date.day_name", "value": "wednesday"}, {"name": "depart_time.period_of_day", "value": "evening"}, {"name": "depart_date.day_name", "value": "thursday"}, {"name": "depart_time.period_of_day", "value": "morning"}]}`
  - parsed: `{"intent": "atis_flight", "slots": [{"name": "fromloc", "value": "toronto"}, {"name": "to", "value": "newark"}, {"name": "departure_date", "value": "wednesday evening or thursday morning"}]}`
  - raw: `{
  "intent": "atis_flight",
  "slots": [
    {
      "name": "fromloc",
      "value": "toronto"
    },
    {
      "name": "to",
      "value": "newark"
    },
    {
      "name": "departure_date",
      "value": "wednesday evening or thursday morning"
    }
  ]
}`

- **id**: `atis-test-6`
  - request: monday morning i would like to fly from columbus to indianapolis
  - gold: `{"intent": "atis_flight", "slots": [{"name": "depart_date.day_name", "value": "monday"}, {"name": "depart_time.period_of_day", "value": "morning"}, {"name": "fromloc.city_name", "value": "columbus"}, {"name": "toloc.city_name", "value": "indianapolis"}]}`
  - parsed: `{"intent": "atis_flight", "slots": [{"name": "departure_date", "value": "2023-11-05"}, {"name": "arrival_date", "value": "2023-11-07"}, {"name": "flight_number", "value": "CA1234"}]}`
  - raw: `{
  "intent": "atis_flight",
  "slots": [
    {
      "name": "departure_date",
      "value": "2023-11-05"
    },
    {
      "name": "arrival_date",
      "value": "2023-11-07"
    },
    {
      "name": "flight_number",
      "value": "CA1234"
    }
  ]
}`

### Malformed (~5)

- **id**: `atis-test-14`
  - request: find flights arriving new york city next saturday
  - gold: `{"intent": "atis_flight", "slots": [{"name": "toloc.city_name", "value": "new york city"}, {"name": "arrive_date.date_relative", "value": "next"}, {"name": "arrive_date.day_name", "value": "saturday"}]}`
  - parsed: `None`
  - raw: `{
  "intent": "find_flights",
  "slots": [
    "city_name", 
    "day_name"
  ]
}`

- **id**: `atis-test-32`
  - request: does tacoma airport offer transportation from the airport to the downtown area
  - gold: `{"intent": "atis_ground_service", "slots": [{"name": "airport_name", "value": "tacoma airport"}]}`
  - parsed: `None`
  - raw: `{
  "intent": "query",
  "slots": [
    "airport_name", 
    "city_name"
  ]
}`

- **id**: `atis-test-111`
  - request: show me all lufthansa flights from seattle to boston with stopovers in minneapolis
  - gold: `{"intent": "atis_flight", "slots": [{"name": "airline_name", "value": "lufthansa"}, {"name": "fromloc.city_name", "value": "seattle"}, {"name": "toloc.city_name", "value": "boston"}, {"name": "stoploc.city_name", "value": "minneapolis"}]}`
  - parsed: `None`
  - raw: `{
  "intent": "show me",
  "slots": [
    "flights",
    "departure_date",
    "arrival_date",
    "stops",
    "mode"
  ]
}`

- **id**: `atis-test-115`
  - request: show me all the morning flights from philadelphia to fort worth
  - gold: `{"intent": "atis_flight", "slots": [{"name": "depart_time.period_of_day", "value": "morning"}, {"name": "fromloc.city_name", "value": "philadelphia"}, {"name": "toloc.city_name", "value": "fort worth"}]}`
  - parsed: `None`
  - raw: `{
  "intent": "show me",
  "slots": [
    "flight",
    "departure_date"
  ]
}`

- **id**: `atis-test-116`
  - request: show me all the flights from kansas city to st. paul
  - gold: `{"intent": "atis_flight", "slots": [{"name": "fromloc.city_name", "value": "kansas city"}, {"name": "toloc.city_name", "value": "st. paul"}]}`
  - parsed: `None`
  - raw: `{
  "intent": "show me",
  "slots": [
    "city_name", 
    "destination"
  ]
}`

### Failure pattern notes

- Malformed / unparsable: 169 (18.9%)
- Wrong intent (parsable): 334
- Wrong slots only: 388
- Correct exact match: 2

