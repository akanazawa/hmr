"""
Module with invoke tasks
"""

import invoke

import photobridge.invoke.host


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(photobridge.invoke.host)
