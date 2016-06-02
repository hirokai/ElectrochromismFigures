import os


def cleanup(task):
    try:
        outs = task.output()
        if isinstance(outs, list):
            for o in outs:
                os.remove(o.path)
        elif isinstance(outs, dict):
            for key, value in outs.iteritems():
                os.remove(value.path)
        else:
            os.remove(outs.path)
    except OSError as e:
        print(e)
