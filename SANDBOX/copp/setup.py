from setuptools import setup, Extension

module = Extension('vish_graphs',
                      sources=['vish_graphs.c'])

setup(name='vish_graphs',
         version='1.0',
         description='Graph generation and manipulation',
         ext_modules=[module])