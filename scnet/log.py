import logging
from accelerate import Accelerator

accelerator = Accelerator()

class MainProcessFilter(logging.Filter):
    def filter(self, record):
        return accelerator.is_main_process
        
log_file = 'training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file) 
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(MainProcessFilter())

torch_logger = logging.getLogger('torch')
torch_logger.setLevel(logging.WARNING)
torch_logger.addFilter(MainProcessFilter())