#!/usr/bin/env python
"""
CR-Learn Command Line Interface
"""
import argparse
import sys
import os
import logging
from cr_learn.CRDS.CRDS_Dashboard import (
    list_datasets, delete_datasets, clear_all_datasets, get_cache_info, DEFAULT_CACHE_DIR
)
from cr_learn.CRDS.CRDS_Health import main as run_health_check

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def show_help():
    """Display detailed help information about all commands."""
    help_text = """
CRLearn 
===============================

CR-Learn provides a Datasets to learn recsys , you can use it for learning.

Available Commands:
------------------

  list dataset       List all downloaded datasets in the cache
  delete dataset     Delete specific datasets from the cache
  clear dataset      Remove all datasets from the cache
  info cache         Show information about the cache directory
  doctor             Run health checks on all dataset preprocessors
  help               Show this help message

Examples:
---------

  cr list dataset                    # List all datasets in cache
  cr delete dataset ml_1m beibei     # Delete specific datasets
  cr clear dataset                   # Clear all datasets (with confirmation)
  cr clear dataset --no-confirm      # Clear all datasets without confirmation
  cr info cache                      # Show cache information
  cr doctor                          # Run health checks on all preprocessors
  cr help                            # Show this help message
  
"""
    print(help_text)
    return 0

def main():
    """Main entry point for the CR-Learn CLI."""
    parser = argparse.ArgumentParser(
        description="CR-Learn Command Line Interface",
        prog="cr"
    )
    
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR,
                        help=f'Path to the cache directory (default: {DEFAULT_CACHE_DIR})')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List resources')
    list_subparsers = list_parser.add_subparsers(dest='resource', help='Resource to list')
    
    # List datasets
    dataset_parser = list_subparsers.add_parser('dataset', help='List all datasets in cache')
    dataset_parser.set_defaults(func=list_datasets)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete resources')
    delete_subparsers = delete_parser.add_subparsers(dest='resource', help='Resource to delete')
    
    # Delete datasets
    delete_dataset_parser = delete_subparsers.add_parser('dataset', help='Delete specified datasets')
    delete_dataset_parser.add_argument('datasets', nargs='+', help='Names of datasets to delete')
    delete_dataset_parser.add_argument('--no-confirm', dest='confirm', action='store_false',
                                      help='Skip confirmation prompt')
    delete_dataset_parser.set_defaults(func=delete_datasets, confirm=True)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear resources')
    clear_subparsers = clear_parser.add_subparsers(dest='resource', help='Resource to clear')
    
    # Clear all datasets
    clear_dataset_parser = clear_subparsers.add_parser('dataset', help='Clear all datasets from cache')
    clear_dataset_parser.add_argument('--no-confirm', dest='confirm', action='store_false',
                                     help='Skip confirmation prompt')
    clear_dataset_parser.set_defaults(func=clear_all_datasets, confirm=True)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information')
    info_subparsers = info_parser.add_subparsers(dest='resource', help='Resource to show information for')
    
    # Cache info
    cache_info_parser = info_subparsers.add_parser('cache', help='Show cache information')
    cache_info_parser.set_defaults(func=get_cache_info)
    
    # Doctor command (health check)
    doctor_parser = subparsers.add_parser('doctor', help='Run health checks on dataset preprocessors')
    doctor_parser.set_defaults(func=lambda args: run_health_check())
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show detailed help information')
    help_parser.set_defaults(func=lambda args: show_help())
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    # Handle the help command
    if args.command == 'help':
        return show_help()
    
    # Check if a command was provided
    if not hasattr(args, 'command') or not args.command:
        parser.print_help()
        return 1
    
    # Special case for doctor command which doesn't need a resource
    if args.command == 'doctor':
        run_health_check()
        return 0
    
    # Check if a resource was provided for commands that need it
    if args.command in ['list', 'delete', 'clear', 'info']:
        if not hasattr(args, 'resource') or not args.resource:
            if args.command == 'list':
                list_parser.print_help()
            elif args.command == 'delete':
                delete_parser.print_help()
            elif args.command == 'clear':
                clear_parser.print_help()
            elif args.command == 'info':
                info_parser.print_help()
            return 1
    
    # Check if a function was assigned
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Execute the function
    args.func(args)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 