#!/usr/bin/env python3
"""
Main entry point for the Fact-Checking Pipeline Server
"""
from app import run_server

if __name__ == '__main__':
    print("Starting Flask server...")
    run_server()
