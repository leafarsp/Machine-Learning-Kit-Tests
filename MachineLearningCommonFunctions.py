import datetime




def print_event_time(str_event):
    t = datetime.datetime.now()
    print(f'{str_event} {t.year:04d}-{t.month:02d}-{t.day:02d} - '
          f'{t.hour:02d}:{t.minute:02d}:{t.second:02d}')