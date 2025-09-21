# SharkGuard
Fake account detection

Instagram Fake Account Detection
Overview
A hackathon project to detect fake Instagram accounts via a web app by analyzing account features with Instagram API data and heuristic rules.

Features
Input Instagram username for analysis

Backed by Instagram API (test/mock data)

Detects likely fake, suspicious, or real accounts based on follower ratios, posts, bio checks, and engagement metrics

Simple web UI for quick real-time feedback

Getting Started
Prerequisites
Node.js + npm (Frontend)

Python 3.x (Backend with Flask)

Instagram API credentials (optional; mock data used by default)

Running Locally
Start backend server: cd backend && flask run

Run frontend: cd ../frontend && npm start

Visit http://localhost:3000

How It Works
Input an Instagram username

Backend fetches/uses mock data

Analysis through rules and simple ML to classify fake accounts

Result displayed with an explanation

Notes
Instagram API public data access may require review, so mock data is used currently

This is a hackathon prototype aimed at demonstrating core functionality
