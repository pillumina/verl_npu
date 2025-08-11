import torch.distributed as dist


old_new_group = None


def monkey_patch_torch_dist():
    print("Applying monkey patch to torch.distributed", flush=True)
    global old_new_group
    if old_new_group is None:
        return 
    
    old_new_group = dist.new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none hccl group.
        if (
            len(args) >= 3 and args[2] == "gloo" or 
            "backend" in kwargs and kwargs["backend"] == "gloo"
        ):
            return group
        
        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))
        
        if len(ranks) == 1:
            return group
        
        group = ReloadableProcessGroup(group, ranks)
        return group
    
    dist.new_group = new_group

    def get_new_function(func):
        def new_function(*args, **kwargs):
            args = (
                arg.group if isinstance(args, ReloadableProcessGroup) else arg
                for arg in args
            )
            kwargs = {
                k: (v.group if isinstance(v, ReloadableProcessGroup) else v)
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)
        return new_function
    
    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)
    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = get_new_function(dist.all_to_all_single)
    dist.broadcast = get_new_function(dist.broadcast)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv
    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg
            
            args = (convert(arg) for arg in args)
            kwargs = {
                k: convert(v)
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)
        return new_function
    
    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup:
    GROUPS = []

    def __init__(self, group, ranks):
        self.group = group
        self.group_info = {
            "ranks": ranks,
        }
        ReloadableProcessGroup.GROUPS.append(self)
    
    def __getatter__(self, name):
        return getattr(self.group, name)
    
    @staticmethod
    def destroy_process_group():
        for reloadable_group in ReloadableProcessGroup.GROUPS:
            if reloadable_group.group is None:
                continue
            dist.destroy_process_group(reloadable_group.group)
            del reloadable_group.group
            reloadable_group.gorup = None

    @staticmethod
    def reload_process_groups():
        for reloadable_group in ReloadableProcessGroup.GROUPS:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(
                ranks=reloadable_group.group_info["ranks"],
            )
            reloadable_group.group = group


def destroy_process_group():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_group()


def reload_process_group():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups
