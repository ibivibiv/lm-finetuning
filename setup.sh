#!/bin/bash

git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=84000'

git config --global user.email "bk@tinymanager.com"
git config --global user.name "Bilal Khan"