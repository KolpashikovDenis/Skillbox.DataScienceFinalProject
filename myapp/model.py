import os
import pickle
import numpy as np

class Model:
    def __init__(self):
        model_path = os.path.join('models', 'forest.pkl')
        source_path = os.path.join('models', 'source.pkl')
        medium_path = os.path.join('models', 'medium.pkl')
        campaign_path = os.path.join('models', 'campaign.pkl')
        adcontent_path = os.path.join('models', 'content.pkl')
        os_path = os.path.join('models', 'os.pkl')
        city_path = os.path.join('models', 'city.pkl')
        
        # your code here
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(source_path, 'rb') as f:
            self.source = pickle.load(f)
        with open(medium_path, 'rb') as f:
            self.medium = pickle.load(f)
        with open(campaign_path, 'rb') as f:
            self.campaign = pickle.load(f)
        with open(adcontent_path, 'rb') as f:
            self.adcontent = pickle.load(f)
        with open(os_path, 'rb') as f:
            self.os = pickle.load(f)
        with open(city_path, 'rb') as f:
            self.city = pickle.load(f)
           


    def predict(self, x):

        x['utm_source']=self.source.loc[self.source['utm_source']==x['utm_source'].iloc[0], 'target_event'].iloc[0]      
        x['utm_medium']=self.medium.loc[self.medium['utm_medium']==x['utm_medium'].iloc[0],'target_event'].iloc[0]        
        x['utm_campaign']=self.campaign.loc[self.campaign['utm_campaign']==x['utm_campaign'].iloc[0],'target_event'].iloc[0]
        x['utm_adcontent']=self.adcontent.loc[self.adcontent['utm_adcontent']==x['utm_adcontent'].iloc[0],'target_event'].iloc[0]
        x['device_os']=self.os.loc[self.os['device_os']==x['device_os'].iloc[0],'target_event'].iloc[0]
        x['geo_city']=self.city.loc[self.city['geo_city']==x['geo_city'].iloc[0],'target_event'].iloc[0]
        
        code = self.model.predict(x[['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'device_os', 'geo_city']])
        return code
