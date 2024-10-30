# DISTRIBUTED TRAINING DDP -> ASHNYCHRONOUS (STATUS : IN PROGRESS)
# DDP has two major types sync and a-sync and it has 4 major blocks 
# setup, cleanup, Parameterserver, worker 


from corerec.common_import import *
from torch.nn import MSELoss
    
import socket

def find_free_port():
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    port = s.getsockname()[1]  # Get the port number
    s.close()
    return port

# def setup(rank, world_size, port=None):
#     """Setup PyTorch environment for distributed training."""
#     if rank == 0 and port is None:
#         port = find_free_port()
#     elif port is None:
#         raise ValueError("Port must be specified for worker processes")

#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = str(port)
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

def setup(rank, world_size, port=None):
    """Setup PyTorch environment for distributed training."""
    print(f"Initializing setup for rank {rank}...")  # Debugging statement

    if rank == 0:
        if port is None:
            port = find_free_port()
            print(f"Master process on rank {rank} using port {port}")  # Debugging statement
    elif port is None:
        raise ValueError("Port must be specified for worker processes")

    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # Initialize the process group
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        print(f"Process group initialized for rank {rank}")  # Debugging statement
    except Exception as e:
        print(f"Failed to initialize process group for rank {rank}: {e}")  # Error handling
        raise
def cleanup():
    """ Cleanup PyTorch distributed environment. """
    dist.destroy_process_group()

class ParameterServer:
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size

    def run(self):
        setup(0, self.world_size)  # Initialize the process group for the parameter server
        # Wait for gradients from each worker (workers are from rank 1 to world_size-1)
        for _ in range(self.world_size - 1):  # Exclude the parameter server itself
            for param in self.model.parameters():
                grad = torch.zeros_like(param.data)
                # Receive gradients from each worker
                dist.recv(tensor=grad, src=dist.get_rank())
                with torch.no_grad():
                    param.data += grad / (self.world_size - 1)  # Average the gradients
        cleanup()  # Clean up the distributed environment after updating the model
loss_function = MSELoss()

def worker(rank, model, data_loader, optimizer, num_epochs, world_size, port):
    setup(rank, world_size, port)  # Initialize the process group for each worker
    for epoch in range(num_epochs):
        for data, target in data_loader:  # Assuming data_loader yields (data, target)
            optimizer.zero_grad()
            # Convert data and target to tensors here if not already
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            # Send gradients to the parameter server
            for param in model.parameters():
                dist.send(tensor=param.grad, dst=0)  # Send gradients to the server at rank 0
    cleanup()  # Clean up after completing the epochs