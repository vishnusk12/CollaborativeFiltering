'''
Created on 27-Nov-2018

@author: Vishnu
'''

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import permission_classes
from rest_framework import permissions
from .model import TopRestaurents, SimilarCuisines

@permission_classes((permissions.AllowAny,))
class Recommend(viewsets.ViewSet):
    def create(self, request):
        question = request.data
        result_rest = TopRestaurents(question["cuisine"])
        result_cuisine = SimilarCuisines(question["cuisine"])
        result_rest.update(result_cuisine)
        return Response(result_rest)