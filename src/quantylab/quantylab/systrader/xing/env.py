import datetime


class Env:
    today = None
    ago5 = None
    ago20 = None
    ago60 = None
    ago120 = None

    today_str = None
    agomonth_str = None
    yesterday_str = None

    @classmethod
    def update(cls):
        if cls.today is not None:
            return
        cls.today = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())
        cls.ago5 = datetime.date.today()-datetime.timedelta(days=5+1)
        cls.ago20 = datetime.date.today()-datetime.timedelta(days=20+1)
        cls.ago60 = datetime.date.today()-datetime.timedelta(days=60+1)
        cls.ago120 = datetime.date.today()-datetime.timedelta(days=120+1)

        cls.today_str = cls.today.strftime('%Y%m%d')
        cls.agomonth_str = (cls.today-datetime.timedelta(weeks=4)).strftime('%Y%m%d')
        cls.yesterday_str = (cls.today-datetime.timedelta(days=1)).strftime('%Y%m%d')

    @classmethod
    def clear(cls):
        if cls.today is None:
            return
        cls.today = None
        cls.ago5 = None
        cls.ago20 = None
        cls.ago60 = None
        cls.ago120 = None

        cls.today_str = None
        cls.agomonth_str = None
        cls.yesterday_str = None

    @classmethod
    def get_today_str(cls):
        if cls.today_str is None:
            cls.update()
        return cls.today_str
