import math
from abc import ABC, abstractmethod

class Student:
    #Create a Python class named Student with attributes name, age, and grade.
    #Include a method display_info() to display the student's information.
    def __init__(self, name, age, grade):
        self.name = name
        self.__age = age
        self.grade = grade

    #Modify the Student class to make the age attribute private. Provide methods
    #set_age() and get_age() to set and retrieve the age of the student.
    def get_age(self):
        return self.__age
    
    def set_age(self, age):
        if age >= 0:
            self.__age = age
        else:
            print("Age cannot be negative.")

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.get_age()}")
        print(f"Grade: {self.grade}")

class HighSchoolStudent(Student):
    #Create a subclass HighSchoolStudent of the Student class. Add an additional
    #attribute grade_level and override the display_info() method to include the grade
    #level.
    def __init__(self, name, age, grade, grade_level):
        # reusing the constructor of the parent class helps 
        # so you don't have to rewrite the code again 
        super().__init__(name, age, grade)
        self.grade_level = grade_level

    def display_info(self):
        super().display_info()
        print(f"Grade Level: {self.grade_level}")



def print_student_info(student):
    #Create a function print_student_info() that accepts an object of either
    #Student or HighSchoolStudent class and prints the student's information using the
    #display_info() method.
    student.display_info()


print_student_info(Student("Alice", 10, "A"))
print_student_info(HighSchoolStudent("Bob", 16, "B", "11th"))

class Shape(ABC):
    #Create an abstract class Shape with an abstract method calculate_area().
    #Implement subclasses Circle and Rectangle inheriting from Shape with methods to
    #calculate their respective areas.
    
    @abstractmethod
    def calculate_area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def calculate_area(self):
        return math.pi * (self.radius ** 2)

class Rectangle(Shape):
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def calculate_area(self):
        return self.height * self.width


print("Circle Area:", Circle(5).calculate_area())
print("Rectangle Area:", Rectangle(10, 5).calculate_area())
    
    