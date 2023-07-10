"""
Module with invoke tasks
"""

import invoke

import photobridge.invoke.host
import photobridge.invoke.visualize


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(photobridge.invoke.host)
ns.add_collection(photobridge.invoke.visualize)
