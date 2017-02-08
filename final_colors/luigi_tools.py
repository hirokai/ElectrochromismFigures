import os
import shutil


def cleanup(task):
    try:
        outs = task.output()
        if isinstance(outs, list):
            for o in outs:
                if os.path.isdir(o.path):
                    shutil.rmtree(o.path)
                else:
                    try:
                        os.remove(o.path)
                    except OSError:
                        pass
        elif isinstance(outs, dict):
            for key, value in outs.iteritems():
                if os.path.isdir(value.path):
                    shutil.rmtree(value.path)
                else:
                    try:
                        os.remove(value.path)
                    except OSError:
                        pass
        else:
            os.remove(outs.path)
    except OSError as e:
        print(e)
