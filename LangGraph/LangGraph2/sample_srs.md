# Software Requirements Specification: RoomBook

## 1. Introduction
RoomBook is a web application that lets university students and staff book study rooms and
meeting rooms in the campus library. It replaces the current paper sign-up sheets at the
library front desk.

## 2. Scope
The system covers searching for rooms, making and cancelling bookings, check-in, and
administration of rooms and booking rules. Payment for rooms is out of scope.

## 3. Users
- **Students**: book study rooms for themselves or a group.
- **Staff**: book meeting rooms, and can book further in advance than students.
- **Library administrators**: manage rooms, opening hours and booking rules.

## 4. Functional Requirements
- **FR-01**: Users must log in with their university single sign-on (SSO) account.
- **FR-02**: Users can search for available rooms by date, time, capacity and equipment
  (e.g. projector, whiteboard).
- **FR-03**: Students can book a room for at most 3 hours per day; staff for at most 8 hours.
- **FR-04**: Students can book up to 7 days in advance; staff up to 30 days in advance.
- **FR-05**: Users can cancel a booking up to 1 hour before it starts.
- **FR-06**: The system sends a confirmation email when a booking is made or cancelled, and a
  reminder 30 minutes before it starts.
- **FR-07**: Users must check in at the room using a QR code within 15 minutes of the start
  time, or the booking is released automatically.
- **FR-08**: Administrators can add, edit and disable rooms, and change opening hours.
- **FR-09**: Administrators can view usage reports showing bookings and no-shows per room.

## 5. Non-Functional Requirements
- **NFR-01**: Search results should load quickly.
- **NFR-02**: The system must support 2,000 concurrent users during exam periods.
- **NFR-03**: The system must be available 99.5% of the time during library opening hours.
- **NFR-04**: Personal data must be stored in compliance with the university's data
  protection policy, and deleted 12 months after a user leaves the university.
- **NFR-05**: The interface must meet WCAG 2.1 AA accessibility standards and work on mobile
  phones.

## 6. Constraints
- The system must integrate with the university's existing SSO provider and email server.
- The first release must be live before the start of the next semester, in 4 months.
- The development team consists of 3 developers.

## 7. Assumptions
- Every room has a fixed QR code poster at its door.
- The SSO provider exposes a standard OAuth 2.0 / OpenID Connect interface.
