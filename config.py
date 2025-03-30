# config.py
class Config:
    def __init__(self):
        self.path = "/Users/ilyatarasevich/Desktop/Goodgame_FASTAPI"
        self.gg_api_url = "https://goodgame.ru/api/4/recommendations/export"
        self.gg_pagination_step = 100
        self.mongo_client_website = 'mongodb://localhost:27017'
        self.model_key = "5a98c0e42e466032ae2a95bfa2d3899e"

config = Config()