from batchgenerators.transforms.abstract_transforms import Compose


class CustomTransform(Compose):
    def __init__(self, transforms):
        super(CustomTransform, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)

        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class CustomTransformVal(CustomTransform):
    def __init__(self, transforms):
        super(CustomTransformVal, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")
