-- 스마트팜 에이전트 AI - MariaDB 스키마
-- 프로토타입은 SQLAlchemy가 테이블을 자동 생성(SQLite 폴백)하지만,
-- 실제 MariaDB 배포 시 아래 스키마를 사용한다.
--
--   mysql -u root -p < db/schema_mariadb.sql

CREATE DATABASE IF NOT EXISTS smartfarm
    DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE smartfarm;

-- 센서/전력/기상 통합 시계열 테이블
CREATE TABLE IF NOT EXISTS sensor_readings (
    ts            DATETIME     NOT NULL COMMENT '측정 시각(10분 간격)',
    appliances    FLOAT        NULL COMMENT '전력 수요(Wh) - 예측 타깃',
    lights        FLOAT        NULL COMMENT '조명 전력(Wh)',
    -- 온실 구역별 온·습도 (zone1~9)
    temp_zone1    FLOAT NULL, humid_zone1 FLOAT NULL,
    temp_zone2    FLOAT NULL, humid_zone2 FLOAT NULL,
    temp_zone3    FLOAT NULL, humid_zone3 FLOAT NULL,
    temp_zone4    FLOAT NULL, humid_zone4 FLOAT NULL,
    temp_zone5    FLOAT NULL, humid_zone5 FLOAT NULL,
    temp_zone6    FLOAT NULL, humid_zone6 FLOAT NULL,
    temp_zone7    FLOAT NULL, humid_zone7 FLOAT NULL,
    temp_zone8    FLOAT NULL, humid_zone8 FLOAT NULL,
    temp_zone9    FLOAT NULL, humid_zone9 FLOAT NULL,
    -- 외부 기상
    temp_out      FLOAT NULL, humid_out  FLOAT NULL,
    pressure      FLOAT NULL, windspeed  FLOAT NULL,
    visibility    FLOAT NULL, dewpoint   FLOAT NULL,
    PRIMARY KEY (ts)
) ENGINE=InnoDB COMMENT='스마트팜 통합 센서/전력/기상 시계열';
