import luigi


class MakeSlicesStub(luigi.Task):
    folder = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.folder)