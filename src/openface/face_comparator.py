###########################
# 
#  this class can compare the similarity between two faces
#
#  typical usage: if there are multiple people in a video, 
#  but we are interested only in one of them,
#  we can compare each face found in the video with a picture of our subject
#
##########################


# TODO: implement get difference score
#    implementation example 
#       def score(person):                  # "distanza tra i vari punti della faccia"
#         a = person.point3D_eyeleft
#         b = person.point3D_eyeright
#         d1 =  eucludean(a,b)
#         a = person.point3D_noseup
#         b = person.point3D_mouthdown
#         d2 = eucludean(a,b)
#         return d1 + d2
#
#      def difference(person1, person2):
#         s1 = score(person1)
#         s2 = score(person2)
#         diff = s1-s2
#         return diff.dot(diff)
#



class FaceComparator():
    """This class is used to compare the similarity between two faces"""

    def __init__(self, data_extractor, partial_path):
        self.partial_path = partial_path
        self.data_extractor = data_extractor
        self.cache = {}

    def get_difference_score(self, faceid, path_to_face):
        """compare how different are the two faces, lower values <--> similar faces"""
        return 1.0

