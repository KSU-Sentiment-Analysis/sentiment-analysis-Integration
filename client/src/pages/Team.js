import React from 'react';
import '../styles/style.css';
import EthanBarnes from '../assets/EthanBarnes.jpg';
import NabeelFaridi from '../assets/NabeelFaridi.jpg';
import IshaMinocha from '../assets/IshaMinocha.jpg';
import ShammahCharles from '../assets/ShammahCharles.jpg';
import AranindIyer from '../assets/AravindIyer.jpg';
import '../assets/role-icons.svg';


const teamMembers = [
    {
        name: 'Ethan Barnes',
        role: 'Lead Developer',
        description: "A deeply passionate computer science professional with a strong focus on machine learning and natural language processing, driven by a desire to build intelligent systems that make a real-world impact",
        image: EthanBarnes,
    },
    {
        name: 'Nabeel Faridi',
        role: 'Developer/Tester',
        description: 'Driven computer science major who played a key role as a Developer and Tester on the project, combining a strong enthusiasm for building reliable software with a keen eye for quality and detail..',
        image: NabeelFaridi,     
    },
    {
        name: 'Shammah Charles',
        role: 'UI/UX Designer',
        description: 'Fueled by a passion for design and user-centered thinking, dedicated to crafting seamless, visually stunning experiences that not only delight users but elevate the entire interaction.',
        image: ShammahCharles,
    },
    {
        name: 'Isha Minocha',
        role: 'UI/UX Designer',
        description: 'A passionate Computer Science major fueled by a love for design and user-centered thinking, dedicated to crafting seamless, visually stunning experiences that not only delight users but elevate the entire interaction.',
        image: IshaMinocha,
    },
    {
        name: 'Aravind lyer',
        role: 'Developer/Tester',
        description: 'A passionate Computer Science major and expert in sentiment analysis and emotion detection algorithms, driven by the goal of bridging human emotion and technology through intelligent systems.',
        image: AranindIyer,
    }
];

const Team = () => {
    return (
        <div className="team-container">
            <h1 className="team-title">Meet Our Team</h1>
            <p className="team-description">
                We're a passionate group of experts dedicated to advancing emotional intelligence in AI.
            </p>
            
            <div className="team-grid">
                {teamMembers.map((member, index) => (
                    <div key={index} className="team-member-card">
                        <div className="team-member-image-container">
                            <img 
                                src={member.image} 
                                alt={member.name}
                                className="team-member-image"
                            />
                            <div className="role-icon-container">
                                <svg className="role-icon" viewBox="0 0 24 24">
                                    <use xlinkHref={`#${member.roleIcon}`}></use>
                                </svg>
                            </div>
                        </div>
                        <h3 className="team-member-name">{member.name}</h3>
                        <h4 className="team-member-role">{member.role}</h4>
                        <p className="team-member-description">{member.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Team;