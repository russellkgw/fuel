class Data::Api::OilFutureFix
  def self.insert_averages
    missing_dates = []
    oil_futures = OilFuture.where("date >= '1995-11-01'").order(date: :asc).all.select { |x| x.date.on_weekday? }

    (Date.new(1995, 11, 1)..(oil_futures.last.date)).each do |date|
      next if date.on_weekend?

      missing_dates << date if oil_futures.select { |x| x.date == date }.empty?
    end

    fixes = {}

    missing_dates.each do |date|
      before = OilFuture.where("date < ?", date).order(date: :desc).first
      after = OilFuture.where("date > ?", date).order(date: :asc).first

      if before.present? && after.present?
        fixes[date.to_s] = (before.settle + after.settle) / 2.0
      end
    end

    fixes.each do |date, settle|
      items = OilFuture.where(date: date)

      OilFuture.create(currency: 'USD',
                       settle: settle,
                       date: date,
                       source: 'AVERAGE-https://www.quandl.com/data/CHRIS/ICE_B1-Brent-Crude-Futures-Continuous-Contract-1-B1-Front-Month',
                       created_at: DateTime.now,
                       updated_at: DateTime.now)
    end

    puts missing_dates.to_s
    puts fixes.to_s
  end

  def self.update_blanks
    blank_oil_futures = OilFuture.where("date >= '1995-11-01'").order(date: :asc)
                      .all.select { |x| x.date.on_weekday? && x.settle.blank? }

    blank_oil_futures.each do |fut|
      before = OilFuture.where("date < ?", fut.date).order(date: :desc).first
      after = OilFuture.where("date > ?", fut.date).order(date: :asc).first

      if before.present? && after.present?
        fut.update({settle: ((before.settle + after.settle) / 2.0)})
        # puts "#{fut.date} - #{((before.settle + after.settle) / 2.0)}"
      end
    end
  end
end