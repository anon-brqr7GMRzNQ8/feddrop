from .sync import Synchronous
from .accel import Accelerate
from .scaffold import Scaffold

session_map = {
    'sync': Synchronous,
    'accel': Accelerate,
    'scaffold':Scaffold,
}
